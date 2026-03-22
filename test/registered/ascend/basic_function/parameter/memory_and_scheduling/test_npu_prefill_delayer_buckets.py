import re
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
    """Test Case: Verify that when the service startup parameters --prefill-delayer-forward-passes-buckets
    and --prefill-delayer-wait-seconds-buckets are configured, querying the /metrics interface shows
    that the statistical groups are consistent with the configuration

    [Test Category] Parameter
    [Test Target] --prefill-delayer-forward-passes-buckets; --prefill-delayer-wait-seconds-buckets
    """

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_0_6B_WEIGHTS_PATH
        cls.forward_passes_buckets = [10.0, 20.0, 30.0]
        cls.wait_seconds_buckets = [1.0, 5.0, 10.0]
        forward_buckets_args = [str(int(b)) for b in cls.forward_passes_buckets]
        wait_seconds_args = [str(int(b)) for b in cls.wait_seconds_buckets]
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
            *forward_buckets_args,
            "--prefill-delayer-wait-seconds-buckets",
            *wait_seconds_args,
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

    # def test_buckets_params(self):
    #     metrics_response = requests.get(f"{DEFAULT_URL_FOR_TEST}/metrics")
    #     self.assertEqual(metrics_response.status_code, 200)
    #     metrics_text = metrics_response.text
    #     # Check that forward passes buckets take effect
    #     for forward_pass in self.forward_passes_buckets:
    #         self.assertIn(f'le="{forward_pass}"', metrics_text)
    #     # Check that wait seconds buckets take effect
    #     for wait_seconds_bucket in self.wait_seconds_buckets:
    #         self.assertIn(f'le="{wait_seconds_bucket}"', metrics_text)

    def _get_metrics_lines(self):
        """辅助方法：获取metrics接口的文本行，按行拆分并过滤空行"""
        try:
            response = requests.get(
                f"{DEFAULT_URL_FOR_TEST}/metrics",
                timeout=10  # 增加请求超时
            )
            response.raise_for_status()  # 主动抛出HTTP错误
            # 拆分并清理行（去除首尾空格、过滤空行）
            return [line.strip() for line in response.text.splitlines() if line.strip()]
        except requests.exceptions.RequestException as e:
            self.fail(f"Failed to fetch metrics: {str(e)}")

    def _check_bucket_in_metric_line(self, metric_name, expected_buckets):
        """
        核心断言方法：精准匹配目标指标行，验证桶配置

        Args:
            metric_name: 目标指标名称（如prefill_delayer_forward_passes_buckets）
            expected_buckets: 期望的桶值列表
        """
        metrics_lines = self._get_metrics_lines()
        target_line = None

        # 1. 筛选目标指标的行（匹配指标名+le标签的行）
        for line in metrics_lines:
            if line.startswith(f"{metric_name}_bucket{{") and 'le="' in line:
                target_line = line
                break

        # 2. 验证是否找到目标行
        self.assertIsNotNone(
            target_line,
            f"Metric line for {metric_name} not found in /metrics response"
        )

        # 3. 提取所有le标签的值并验证
        # 正则匹配所有le="数值"的部分
        le_matches = re.findall(r'le="([\d\.]+)"', target_line)
        le_values = [float(v) for v in le_matches]

        # 4. 验证期望的桶值都存在于实际值中
        for bucket in expected_buckets:
            self.assertIn(
                bucket,
                le_values,
                f"Expected bucket {bucket} not found in {metric_name} (actual: {le_values})"
            )

    def test_buckets_params(self):
        """Test bucket parameters take effect accurately"""
        # 验证forward passes桶配置
        self._check_bucket_in_metric_line(
            "prefill_delayer_forward_passes_buckets",
            self.forward_passes_buckets
        )

        # 验证wait seconds桶配置
        self._check_bucket_in_metric_line(
            "prefill_delayer_wait_seconds_buckets",
            self.wait_seconds_buckets
        )


if __name__ == "__main__":
    unittest.main()
