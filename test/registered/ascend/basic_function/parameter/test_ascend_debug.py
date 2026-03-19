import os
import unittest

import requests

from types import SimpleNamespace

from sglang.test.run_eval import run_eval

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=50, suite="nightly-1-npu-a3", nightly=True)


class TestSkipServerWarmup(CustomTestCase):
    """
    Testcase：Verify that if --skip-server-warmup parameter set, skip warmup.

    [Test Category] Parameter
    [Test Target] --skip-server-warmup
    """

    @classmethod
    def setUpClass(cls):
        cls.model_path = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--skip-server-warmup",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
        ]

        cls.out_log_file = open("./warmup_out_log.txt", "w+", encoding="utf-8")
        cls.err_log_file = open("./warmup_err_log.txt", "w+", encoding="utf-8")
        cls.process = popen_launch_server(
            cls.model_path,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        cls.err_log_file.close()
        os.remove("./warmup_out_log.txt")
        os.remove("./warmup_err_log.txt")

    # def test_skip_server_warmup(self):
    #     response = requests.post(
    #         f"{self.base_url}/generate",
    #         json={
    #             "text": "The capital of France is",
    #             "sampling_params": {"temperature": 0, "max_new_tokens": 32},
    #         },
    #     )
    #     # Verify that inference is correct when warming up is skipped
    #     self.assertEqual(response.status_code, 200)
    #     self.assertIn("Paris", response.text)
    #     self.out_log_file.seek(0)
    #
    #     # warm up will send a GET /get_model_info request and a generate request to warm up server.
    #     content = self.out_log_file.read()
    #     self.assertTrue(len(content) > 0)
    #     self.assertNotIn("GET /model_info HTTP/1.1", content)

    def test_mgsm_en(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model_path,
            eval_name="mgsm_en",
            num_examples=10,
            num_threads=1024,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.5)


if __name__ == "__main__":
    unittest.main()
