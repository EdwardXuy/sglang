import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestNpuPrefillDelayerBuckets(CustomTestCase):
    """Test Case: Verify the accuracy of LLM models under TP+PP hybrid parallelism

    [Test Category] Parameter
    [Test Target] --pp-size; --tp-size
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_0_6B_WEIGHTS_PATH
        other_args = [
            "--tp-size",
            "2",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--enable-metrics",
            "--enable-prefill-delayer",
            "--enable-dp-attention",
            "--dp-size",
            "2",
            "--prefill-delayer-max-delay-passes",
            "100",
            "--prefill-delayer-forward-passes-buckets",
            "10 20 30",
            "--prefill-delayer-wait-seconds-buckets",
            "1 5 10",
        ]
        cls.process = popen_launch_server(
            cls.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_buckets_params(self):
        # # 1. 启动带 buckets 参数的服务
        # process = _launch_server(
        #     model="Qwen/Qwen3-0.6B",
        #     base_url=DEFAULT_URL_FOR_TEST,
        #     prefill_delayer=True,
        #     other_args=[
        #         "--prefill-delayer-max-delay-passes", "100",
        #         "--prefill-delayer-forward-passes-buckets", "10", "20", "30",
        #         "--prefill-delayer-wait-seconds-buckets", "0.1", "0.5", "1.0",
        #         "--max-total-tokens", "50000",
        #     ],
        # )
        #
        # # 2. 发送测试请求（长请求+短请求）
        # async def send_test_requests():
        #     client = openai.AsyncClient(base_url=f"{DEFAULT_URL_FOR_TEST}/v1", api_key="EMPTY")
        #     # 发长请求
        #     asyncio.create_task(client.chat.completions.create(
        #         model="Qwen/Qwen3-0.6B",
        #         messages=[{"role": "user", "content": "Hello " * 5000}],
        #         max_tokens=10000,
        #         extra_body={"data_parallel_rank": 0},
        #     ))
        #     await asyncio.sleep(2)
        #     # 发100个短请求
        #     for i in range(100):
        #         await client.chat.completions.create(
        #             model="Qwen/Qwen3-0.6B",
        #             messages=[{"role": "user", "content": f"Say hi {i}"}],
        #             max_tokens=10,
        #             extra_body={"data_parallel_rank": 1},
        #         )
        #
        # asyncio.run(send_test_requests())

        # 3. 查监控指标，验证 buckets 生效
        metrics_text = _print_prefill_delayer_metrics(DEFAULT_URL_FOR_TEST, expect_metrics=True)
        # 检查轮次 buckets
        assert 'le="10"' in metrics_text and 'le="20"' in metrics_text and 'le="30"' in metrics_text
        # 检查时间 buckets
        assert 'le="1"' in metrics_text and 'le="5"' in metrics_text and 'le="10"' in metrics_text


def _print_prefill_delayer_metrics(base_url: str, expect_metrics: bool) -> str:
    metrics_response = requests.get(f"{base_url}/metrics")
    assert metrics_response.status_code == 200
    metrics_text = metrics_response.text
    prefill_delayer_metrics = [
        line for line in metrics_text.split("\n") if "prefill_delayer" in line
    ]
    print("=== PrefillDelayer Metrics ===")
    for line in prefill_delayer_metrics:
        print(line)
    if expect_metrics:
        assert "sglang:prefill_delayer_wait_forward_passes" in metrics_text
        assert "sglang:prefill_delayer_wait_seconds" in metrics_text
        assert "sglang:prefill_delayer_outcomes_total" in metrics_text
    return metrics_text

if __name__ == "__main__":
    unittest.main()
