import io
import unittest
import time
import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class BaseTestWatchdog:
    env_override = None
    expected_crash_message = None
    process = None

    @classmethod
    def setUpClass(cls):
        cls.stdout = io.StringIO()
        cls.stderr = io.StringIO()

        # 模拟指定模块阻塞，启动服务并设置看门狗
        with cls.env_override():
            try:
                cls.process = popen_launch_server(
                    QWEN3_0_6B_WEIGHTS_PATH,
                    DEFAULT_URL_FOR_TEST,
                    timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                    other_args=[
                        "--watchdog-timeout",
                        20,
                        "--skip-server-warmup",
                    ],
                    return_stdout_stderr=(cls.stdout, cls.stderr),
                )
                time.sleep(cls.watchdog_timeout + 5)
            except Exception as e:
                print(f"Service launch exception (expected for watchdog): {e}")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.stdout.close()
        cls.stderr.close()

    def test_watchdog_crashes_server(self):
        """核心测试：验证看门狗触发后服务崩溃"""

        print("Start call /generate API", flush=True)
        try:
            response = requests.post(
                DEFAULT_URL_FOR_TEST + "/generate",
                json={
                    "text": "Hello, please repeat this sentence for 1000 times.",
                    "sampling_params": {"max_new_tokens": 100, "temperature": 0},
                },
                timeout=30,
            )
            # 如果请求成功，说明服务未崩溃，测试失败
            self.fail(f"API request succeeded unexpectedly, server should be crashed")
        except requests.exceptions.ConnectionError:
            # 预期的连接失败，说明服务已崩溃
            print("API request failed (expected): Server is crashed as watchdog triggered")

        # 日志中包含服务崩溃关键词
        combined_output = self.stdout.getvalue() + self.stderr.getvalue()
        self.assertIn(
            self.expected_crash_message,
            combined_output,
            f"Expected crash message '{self.expected_crash_message}' not found in logs"
        )
        print(f"Verified: Found crash message '{self.expected_crash_message}' in logs")

        # # 验证进程已退出（看门狗触发崩溃）
        # self.assertIsNotNone(
        #     self.process.poll(),
        #     f"Process should exit after watchdog timeout ({self.watchdog_timeout}s), but still running"
        # )


class TestWatchdogDetokenizer(BaseTestWatchdog, CustomTestCase):
    # Detokenizer阻塞触发看门狗超时，服务崩溃
    env_override = lambda: envs.SGLANG_TEST_STUCK_DETOKENIZER.override(30)
    expected_crash_message = "DetokenizerManager watchdog timeout, crashing server to prevent hanging"


class TestWatchdogTokenizer(BaseTestWatchdog, CustomTestCase):
    # Tokenizer阻塞触发看门狗超时，服务崩溃
    env_override = lambda: envs.SGLANG_TEST_STUCK_TOKENIZER.override(30)
    expected_crash_message = "TokenizerManager watchdog timeout, crashing server to prevent hanging"


class TestWatchdogSchedulerInit(BaseTestWatchdog, CustomTestCase):
    # Scheduler初始化阻塞触发看门狗超时，服务崩溃
    env_override = lambda: envs.SGLANG_TEST_STUCK_SCHEDULER_INIT.override(30)
    expected_crash_message = "Scheduler watchdog timeout, crashing server to prevent hanging"


if __name__ == "__main__":
    unittest.main()
