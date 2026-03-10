import json
import logging
import unittest
import tempfile
import os

import requests

from sglang.srt.utils import kill_process_tree

from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=600, suite="nightly-1-npu-a3", nightly=True)


class TestApiRelatedApiKey(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "test-api-key-12345"
        cls.custom_model_name = "Llama3.2"
        cls.weight_version = "v1.0.0"
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--api-key",
            cls.api_key,
            "--served-model-name",
            cls.custom_model_name,
            "--weight-version",
            cls.weight_version,
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    def test_served_model_weight_version(self):
        response = requests.get(f"{self.base_url}/v1/models")
        result = response.json()

        self.assertIn("data", result)
        self.assertEqual(result["data"][0]["id"], self.custom_model_name)
        self.assertEqual(result["data"][0]["weight_version"], self.weight_version)

        logging.warning(f"Request with api-key auth succeeded: {result['text'][:50]}")
        logging.warning(f"Weight version works: {result['text'][:50]}")

    def test_api_key_auth(self):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "Explain neural networks:",
                "sampling_params": {
                    "max_new_tokens": 64,
                },
            },
            headers = headers,
        )
        result = response.json()

        self.assertIn("text", result)
        self.assertGreater(len(result["text"]), 0)
        logging.warning(f"Request with api-key auth succeeded: {result['text'][:50]}")


class TestApiRelatedAdminApiKey(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "test-api-key-12345"
        cls.custom_model_name = "Llama3.2"
        cls.weight_version = "v1.0.0"
        other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--api-key",
            cls.api_key,
            "--served-model-name",
            cls.custom_model_name,
            "--weight-version",
            cls.weight_version,
        ]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    def test_served_model_weight_version(self):
        response = requests.get(f"{self.base_url}/v1/models")
        result = response.json()

        self.assertIn("data", result)
        self.assertEqual(result["data"][0]["id"], self.custom_model_name)
        self.assertEqual(result["data"][0]["weight_version"], self.weight_version)

        logging.warning(f"Request with api-key auth succeeded: {result['text'][:50]}")
        logging.warning(f"Weight version works: {result['text'][:50]}")

    def test_api_key_auth(self):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "Explain neural networks:",
                "sampling_params": {
                    "max_new_tokens": 64,
                },
            },
            headers = headers,
        )
        result = response.json()

        self.assertIn("text", result)
        self.assertGreater(len(result["text"]), 0)
        logging.warning(f"Request with api-key auth succeeded: {result['text'][:50]}")




