import os
import subprocess
import sys
import unittest
from types import SimpleNamespace

from huggingface_hub import hf_hub_download

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    FR_SPEC_TOKEN_MAP_PATH,
    LLAMA_3_8B_EAGLE_WEIGHTS_PATH,
    LLAMA_3_8B_INSTRUCT_WEIGHTS_PATH,
    QWEN3_32B_EAGLE3_WEIGHTS_PATH,
    QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=300, suite="nightly-8-npu-a3", nightly=True)

CI_CACHE_DIR = os.environ.get("CI_CACHE_DIR", "/tmp/sglang_test_cache")
FR_SPEC_TOKEN_MAP_REPO = "thunlp/LLaMA3-Instruct-8B-FR-Spec"
FR_SPEC_TOKEN_MAP_FILENAME = "freq_32768.pt"
FR_SPEC_TOKEN_MAP_LOCAL_PATH = os.path.join(CI_CACHE_DIR, FR_SPEC_TOKEN_MAP_FILENAME)

def ensure_package(package_name):
    """确保 Python 包已安装，若没有则自动安装。"""
    try:
        __import__(package_name)
    except ImportError:
        print(f"Package '{package_name}' not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package_name])

def download_with_hf_hub(endpoint, cache_dir):
    return hf_hub_download(
        repo_id=FR_SPEC_TOKEN_MAP_REPO,
        filename=FR_SPEC_TOKEN_MAP_FILENAME,
        cache_dir=cache_dir,
        endpoint=endpoint,
        resume_download=True,
    )

def download_with_wget(url, dest_path):
    """使用 wget 直接下载文件。"""
    subprocess.check_call(["wget", "-O", dest_path, url])


def ensure_token_map_downloaded():
    """确保 token map 文件已下载到本地，并返回本地路径。"""
    os.makedirs(CI_CACHE_DIR, exist_ok=True)

    if os.path.exists(FR_SPEC_TOKEN_MAP_LOCAL_PATH):
        print(f"Token map file already exists at: {FR_SPEC_TOKEN_MAP_LOCAL_PATH}")
        return FR_SPEC_TOKEN_MAP_LOCAL_PATH

    # 1. 确保 huggingface_hub 已安装
    try:
        ensure_package("huggingface_hub")
    except Exception as e:
        print(f"Failed to install huggingface_hub: {e}. Will fallback to wget.")

    # 2. 定义要尝试的端点列表（镜像优先）
    endpoints = [
        ("HF Mirror", "https://hf-mirror.com"),
        ("HF Official", "https://huggingface.co"),
    ]

    # 3. 首先尝试用 huggingface_hub 下载
    for name, endpoint in endpoints:
        try:
            print(f"Attempting download via {name} ({endpoint})...")
            path = download_with_hf_hub(endpoint, CI_CACHE_DIR)
            print(f"Download succeeded from {name}. File at: {path}")
            return path
        except Exception as e:
            print(f"Download from {name} failed: {e}")

    # 4. huggingface_hub 方式全部失败，尝试直接用 wget 下载镜像上的原始文件
    print("All huggingface_hub attempts failed. Trying direct wget from HF Mirror...")
    fallback_url = f"https://hf-mirror.com/{FR_SPEC_TOKEN_MAP_REPO}/resolve/main/{FR_SPEC_TOKEN_MAP_FILENAME}"
    try:
        download_with_wget(fallback_url, FR_SPEC_TOKEN_MAP_LOCAL_PATH)
        print(f"Download succeeded via wget. File at: {FR_SPEC_TOKEN_MAP_LOCAL_PATH}")
        return FR_SPEC_TOKEN_MAP_LOCAL_PATH
    except Exception as e:
        raise RuntimeError(f"Failed to download token map file using all methods. Last error: {e}")

class TestNpuSpeculativeTokenMap(CustomTestCase):
    """Test --speculative-token-map with EAGLE3 (ignored) and EAGLE (enabled).

    Both cases run GSM8K evaluation to ensure accuracy does not degrade.
    """

    def test_eagle3_ignores_token_map_gsm8k(self):
        """EAGLE3 ignores token map; GSM8K accuracy should meet threshold."""
        args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--quantization",
            "modelslim",
            "--disable-radix-cache",
            "--speculative-algorithm",
            "EAGLE3",
            "--speculative-draft-model-path",
            QWEN3_32B_EAGLE3_WEIGHTS_PATH,
            "--speculative-draft-model-quantization",
            "unquant",
            "--speculative-num-steps",
            "4",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "5",
            "--speculative-attention-mode",
            "decode",
            "--speculative-token-map",
            "/nonexistent/token_map.pt",  # ignored
            "--tp-size",
            "8",
            "--mem-fraction-static",
            "0.7",
            "--disable-cuda-graph",
            "--dtype",
            "bfloat16",
        ]
        env = os.environ.copy()
        env.update(
            {
                "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
                "SGLANG_ENABLE_SPEC_V2": "1",
            }
        )
        process = popen_launch_server(
            QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3,
            other_args=args,
            env=env,
        )
        try:
            eval_args = SimpleNamespace(
                base_url=DEFAULT_URL_FOR_TEST,
                eval_name="gsm8k",
                api="completion",
                num_examples=1319,
                num_threads=128,
                max_new_tokens=512,
                num_shots=5,
                temperature=0.0,
            )
            metrics = run_eval(eval_args)
            self.assertGreaterEqual(metrics["score"], 0.86)
        finally:
            kill_process_tree(process.pid)

    def test_eagle_with_valid_token_map_gsm8k(self):
        """EAGLE (EAGLE-2) with valid token map; GSM8K accuracy should meet threshold."""

        local_token_map_path = ensure_token_map_downloaded()

        args = [
            "--trust-remote-code",
            "--attention-backend",
            "ascend",
            "--disable-radix-cache",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-draft-model-path",
            LLAMA_3_8B_EAGLE_WEIGHTS_PATH,
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "4",
            "--speculative-num-draft-tokens",
            "16",
            "--speculative-token-map",
            local_token_map_path,
            "--tp-size",
            "4",
            "--mem-fraction-static",
            "0.7",
            "--disable-cuda-graph",
            "--dtype",
            "float16",
        ]
        env = os.environ.copy()
        env.update(
            {
                "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
                "SGLANG_ENABLE_SPEC_V2": "1",
            }
        )
        process = popen_launch_server(
            LLAMA_3_8B_INSTRUCT_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3,
            other_args=args,
            env=env,
        )
        try:
            eval_args = SimpleNamespace(
                base_url=DEFAULT_URL_FOR_TEST,
                eval_name="gsm8k",
                api="completion",
                num_examples=1319,
                num_threads=128,
                max_new_tokens=512,
                num_shots=5,
                temperature=0.0,
            )
            metrics = run_eval(eval_args)
            self.assertGreaterEqual(
                metrics["score"], 0.79
            )  # adjust threshold as needed
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
