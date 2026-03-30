import unittest
import time

from sglang.test.ascend.test_ascend_utils import PHI_4_MULTIMODAL_INSTRUCT_WEIGHTS_PATH
from sglang.test.ascend.vlm_utils import TestVLMModels
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=800, suite="nightly-4-npu-a3", nightly=True)

class TestPhi4MultimodalCompareLatency(TestVLMModels):
    """
    Testcase: Compare latency with/without --keep-mm-feature-on-device.
    Verify latency decreases when enabling --keep-mm-feature-on-device.
    """

    model = PHI_4_MULTIMODAL_INSTRUCT_WEIGHTS_PATH
    mmmu_accuracy = 0.2

    # 基础公共参数
    base_args = [
        "--trust-remote-code",
        "--mem-fraction-static", "0.8",
        "--attention-backend", "ascend",
        "--disable-cuda-graph",
        "--disable-radix-cache",
    ]

    def _run_test_with_args(self, extra_args):
        """运行测试并返回 metrics（包含 latency）"""
        # 拼接完整启动参数
        self.other_args = self.base_args + extra_args
        # 执行测试
        metrics = self._run_vlm_mmmu_test()
        return metrics

    def test_latency_compare_keep_mm_feature_on_device(self):
        """
        对比两组配置：
        1. 不带 --keep-mm-feature-on-device
        2. 带 --keep-mm-feature-on-device
        验证带参数时 latency 更低
        """
        print("\n===== 开始对比延迟：with / without --keep-mm-feature-on-device =====")

        # ----------------------
        # 1. 不带参数
        # ----------------------
        print("\n[测试 1] 不带 --keep-mm-feature-on-device")
        metrics_off = self._run_test_with_args([])
        latency_off = metrics_off["latency"]  # 根据实际返回的 key 调整，比如 avg_latency
        print(f"延迟结果: {latency_off:.4f} s")

        # ----------------------
        # 2. 带参数
        # ----------------------
        print("\n[测试 2] 带 --keep-mm-feature-on-device")
        metrics_on = self._run_test_with_args([
            "--keep-mm-feature-on-device"
        ])
        latency_on = metrics_on["latency"]  # 根据实际返回的 key 调整
        print(f"延迟结果: {latency_on:.4f} s")

        # ----------------------
        # 对比结果
        # ----------------------
        print("\n===== 最终对比结果 =====")
        print(f"不带参数延迟: {latency_off:.4f}")
        print(f"带参数延迟  : {latency_on:.4f}")

        # 断言：带参数延迟必须更低
        self.assertLess(latency_on, latency_off,
            f"开启 --keep-mm-feature-on-device 后延迟没有降低！"
            f"on={latency_on:.4f}, off={latency_off:.4f}")

        print("\n✅ 测试通过：开启 --keep-mm-feature-on-device 有效降低延迟！")

if __name__ == "__main__":
    unittest.main()

