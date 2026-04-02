"""
python3 -m unittest openai_server.validation.test_large_max_new_tokens.TestLargeMaxNewTokens.test_chat_completion
"""

import os
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import openai

from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.ascend.output_capturer import OutputCapturer
from sglang.test.ascend.test_npu_logging import TestNPULoggingBase
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci, register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    STDERR_FILENAME,
    STDOUT_FILENAME,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=41, suite="stage-b-test-1-gpu-large")
register_amd_ci(est_time=41, suite="stage-b-test-1-gpu-small-amd")
register_npu_ci(est_time=100, suite="nightly-1-npu-a3", nightly=True)


class TestNPUEnableRequestTimeStatsLogging(TestNPULoggingBase):
    """Testcase: Verify the functionality of --enable-request-time-stats-logging to generate Req Time Stats logs on Ascend backend with Llama-3.2-1B-Instruct model.

    [Test Category] Parameter
    [Test Target] --enable-request-time-stats-logging
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.output_capturer = OutputCapturer()
        cls.output_capturer.start()
        cls.other_args.extend(["--decode-log-interval", "2",])
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
            env={"SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION": "256", **os.environ},
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls.output_capturer.stop()

    def inference(self):
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "Please repeat the world 'hello' for 10000 times.",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 10000,
                },
            },
        )

        self.assertEqual(response.status_code, 200, "Failed to call generate API")
        return response

    def test_enable_request_time_stats_logging(self):



        num_requests = 4

        futures = []
        with ThreadPoolExecutor(num_requests) as executor:
            # Send multiple requests
            for i in range(num_requests):
                futures.append(executor.submit(self.inference()))


        self.assertIn(f"#running-req: {num_requests}", self.output_capturer.get_all())




if __name__ == "__main__":
    unittest.main()