import os
import unittest
import requests
from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_32B_EAGLE3_WEIGHTS_PATH,
    QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH,
    LLAMA_3_8B_INSTRUCT_WEIGHTS_PATH,
    LLAMA_3_8B_EAGLE_WEIGHTS_PATH,
    FR_SPEC_TOKEN_MAP_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=300, suite="nightly-8-npu-a3", nightly=True)


class TestNpuSpeculativeTokenMap(CustomTestCase):
    """Test --speculative-token-map behavior in EAGLE3 and EAGLE (EAGLE-2).

    - EAGLE3: parameter should be ignored (even with invalid path), server starts
      and inference works normally.
    - EAGLE (EAGLE-2): parameter should accept a valid .pt token map file from HF,
      enabling FR-Spec optimization. If the required models or file are missing,
      the test is skipped.

    [Test Category] Parameter
    [Test Target] --speculative-token-map
    """

    def test_eagle3_ignores_token_map(self):
        """EAGLE3 ignores --speculative-token-map; even invalid path should not break."""
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
            "/nonexistent/token_map.pt",  # invalid path
            "--tp-size",
            "8",
            "--mem-fraction-static",
            "0.7",
            "--disable-cuda-graph",
            "--dtype",
            "bfloat16",
        ]
        env = os.environ.copy()
        env.update({
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            "SGLANG_ENABLE_SPEC_V2": "1",
        })
        process = popen_launch_server(
            QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3,
            other_args=args,
            env=env,
        )
        try:
            # Verify health and simple inference
            health = requests.get(f"{DEFAULT_URL_FOR_TEST}/health", timeout=10)
            self.assertEqual(health.status_code, 200)

            resp = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/v1/chat/completions",
                json={
                    "model": QWEN3_32B_W8A8_MINDIE_WEIGHTS_PATH,
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "max_tokens": 16,
                },
                timeout=60,
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("choices", data)
            self.assertTrue(len(data["choices"][0]["message"]["content"]) > 0)
        finally:
            kill_process_tree(process.pid)

    def test_eagle_with_valid_token_map(self):
        """EAGLE (EAGLE-2) with valid token map file from HF should start and infer."""
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
            FR_SPEC_TOKEN_MAP_PATH,   # HF path, auto-download
            "--tp-size",
            "4",
            "--mem-fraction-static",
            "0.7",
            "--disable-cuda-graph",
            "--dtype",
            "float16",
        ]
        env = os.environ.copy()
        env.update({
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            "SGLANG_ENABLE_SPEC_V2": "1",
        })
        process = popen_launch_server(
            LLAMA_3_8B_INSTRUCT_WEIGHTS_PATH,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3,
            other_args=args,
            env=env,
        )
        try:
            health = requests.get(f"{DEFAULT_URL_FOR_TEST}/health", timeout=10)
            self.assertEqual(health.status_code, 200)

            prompt = "What is the capital of France?"
            resp = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/v1/chat/completions",
                json={
                    "model": LLAMA_3_8B_INSTRUCT_WEIGHTS_PATH,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 64,
                    "temperature": 0,
                },
                timeout=120,
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("choices", data)
            content = data["choices"][0]["message"]["content"]
            self.assertGreater(len(content.strip()), 0)

            self.assertIn(
                "paris",
                content.lower(),
                f"Expected 'Paris' in response, but got: {content[:200]}",
            )

            print(f"Q: {prompt}")
            print(f"A: {content}")
        finally:
            kill_process_tree(process.pid)


if __name__ == "__main__":
    unittest.main()
