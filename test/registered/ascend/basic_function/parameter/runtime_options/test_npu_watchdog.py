import os
import unittest

from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestWatchdogTimeout(CustomTestCase):
    """Testcase:

    [Test Category] Parameter
    [Test Target] --watchdog-timeout
    """

    def test_watchdog_timeout(self):
        expected_timeout_message = "Scheduler watchdog timeout (self.watchdog_timeout=1.0, self.soft=False)"
        expected_crash_message = "SIGQUIT received."
        out_log_file = open("./cache_out_log.txt", "w+", encoding="utf-8")
        err_log_file = open("./cache_err_log.txt", "w+", encoding="utf-8")
        try:
            popen_launch_server(
                QWEN3_0_6B_WEIGHTS_PATH,
                DEFAULT_URL_FOR_TEST,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--watchdog-timeout",
                    1,
                    "--skip-server-warmup",
                    "--attention-backend",
                    "ascend",
                ],
                return_stdout_stderr=(out_log_file, err_log_file),
            )
        except Exception as e:
            print(f"Server launch failed as expects:{e}")
        finally:
            err_log_file.seek(0)
            content = err_log_file.read()
            self.assertIn(
                expected_timeout_message,
                content,
                f"Expected timeout message '{expected_timeout_message}' not found in logs"
            )
            self.assertIn(
                expected_crash_message,
                content,
                f"Expected crash message '{expected_crash_message}' not found in logs"
            )
            out_log_file.close()
            err_log_file.close()
            os.remove("./cache_out_log.txt")
            os.remove("./cache_err_log.txt")


if __name__ == "__main__":
    unittest.main()
