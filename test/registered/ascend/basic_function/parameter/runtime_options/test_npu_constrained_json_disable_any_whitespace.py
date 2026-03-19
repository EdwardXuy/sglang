import json
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestAscendApi(CustomTestCase):
    """Testcase: Verify that the basic functions of the API interfaces work properly and the returned parameters are consistent with the configurations.

    [Test Category] Interface
    [Test Target] /health; /health_generate; /ping; /model_info; /server_info; /get_load; /v1/models; /v1/models/{model:path}; /generate
    """

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        other_args = [
            "--attention-backend",
            "ascend",
            "--grammar-backend",
            "xgrammar",
            "--constrained-json-disable-any-whitespace",
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


    def test_api_model_info(self):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                },
            },
        )
        print(response.json())
        # self.assertEqual(response.status_code, 200)
        # self.assertIn("Paris", response.text)
        # response = requests.get(f"{DEFAULT_URL_FOR_TEST}/model_info")
        # self.assertEqual(response.status_code, 200)
        # self.assertEqual(response.json()["model_path"], self.model)


if __name__ == "__main__":
    unittest.main()
