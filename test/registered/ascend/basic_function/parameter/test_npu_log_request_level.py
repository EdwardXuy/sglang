import os
import re
import tempfile
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    popen_launch_server, CustomTestCase, DEFAULT_URL_FOR_TEST,
)

register_npu_ci(est_time=100, suite="nightly-1-npu-a3", nightly=True)


class TestNPULogRequestLevel0(CustomTestCase):
    """Test case class for verifying the --log-requests-level=0 parameter on Ascend NPU.

    This test class validates that the inference server correctly logs request metadata
    at verbosity level 0 when the --log-requests and --log-requests-level flags are enabled.

    [Test Category] Parameter
    [Test Target] --log-requests; --log-requests-level
    """

    log_requests_level = 0

    @classmethod
    def setUpClass(cls):
        cls.model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.other_args = [
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.8",
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--log-requests",
            "--log-requests-level",
            str(cls.log_requests_level),
        ]
        cls.out_log_file_obj = tempfile.NamedTemporaryFile(
            mode="w+", encoding="utf-8", delete=False, suffix=".txt"
        )
        cls.out_log_name = cls.out_log_file_obj.name
        cls.out_log_file = cls.out_log_file_obj
        cls.err_log_file_obj = tempfile.NamedTemporaryFile(
            mode="w+", encoding="utf-8", delete=False, suffix=".txt"
        )
        cls.err_log_name = cls.err_log_file_obj.name
        cls.err_log_file = cls.err_log_file_obj

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.other_args,
            return_stdout_stderr=(cls.out_log_file, cls.err_log_file),
        )

    def test_log_requests_level(self, log_requests_level, out_log_file):
        """
        Validate that log content complies with expectations for different --log-requests-level configurations.

        Core Functionality:
            1. Send a request to the model to generate the longest possible string, with token generation limits optimized for efficiency:
               - Max 100 new tokens for --log-requests-level ≤ 1 (reduce generation time for low-detail logging)
               - Max 2500 new tokens for --log-requests-level ≥ 2 (exceeds 2048 to test truncation behavior)
            2. Verify the log file contains level-specific keywords matching the target log_requests_level
            3. Validate token count preservation rules in logs:
               - Level 2: Logs are truncated to retain only 2048 tokens (partial input/output)
               - Level 3: Logs retain all generated tokens (full input/output)
               - Levels ≤1: No token count validation (only metadata/sampling params logged)

        Args:
            log_requests_level (int): Target log verbosity level (0/1/2/3) for validation; maps to the
                --log-requests-level parameter
            out_log_file (file object): Open file object of the out log file
        """
        # Step 1: Send a request to the model to generate the longest possible string, with token generation limits optimized for efficiency:
        max_new_token = 2500 if log_requests_level >= 2 else 100
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": f"just return me a long string, generate as much as possible.",
                "sampling_params": {"temperature": 0, "max_new_tokens": max_new_token},
            },
        )
        self.assertEqual(response.status_code, 200)

        # Step 2: Verify the log file contains level-specific keywords matching the target log_requests_level
        out_log_file.seek(0)
        content = out_log_file.read()
        self.assertTrue(len(content) > 0)
        self.assertIsNotNone(
            re.search(self.log_request_message_dict[str(log_requests_level)], content)
        )
        # The total number of generated tokens should equal the configured maximum number of generated tokens
        lines = self.get_lines_with_keyword(self.out_log_name, self.keyword_Finish)
        self.assertGreater(len(lines), 0, "Did not find finish message in log.")
        finish_message = lines[-1]["content"]
        self.assertIn(f"'completion_tokens': {max_new_token}", finish_message)

        # Step 3: Validate token count preservation rules in logs:
        if log_requests_level >= 2:
            # Extract the content of output_ids to count the number of generated tokens recorded in the logs
            output_ids_start_index = finish_message.find(
                self.keyword_output_id_start
            ) + len(self.keyword_output_id_start)
            output_ids_end_index = finish_message.find(self.keyword_output_id_end)
            output_ids_list_str = finish_message[
                output_ids_start_index:output_ids_end_index
            ].strip()
            if log_requests_level == 2:
                # When --log-requests-level=2, the log records a maximum of 2048 tokens (truncated content)
                self.assertIn("] ... [", output_ids_list_str)
                output_ids_list_str = output_ids_list_str.replace("] ... [", ", ")
                token_id_count = len(
                    [
                        x.strip()
                        for x in re.split(r",\s*", output_ids_list_str)
                        if x.strip()
                    ]
                )
                self.assertTrue(token_id_count == 2048)
            else:
                # When --log-requests_level=3, the log records all generated token content (no truncation)
                token_id_count = len(
                    [
                        x.strip()
                        for x in re.split(r",\s*", output_ids_list_str)
                        if x.strip()
                    ]
                )
                self.assertTrue(token_id_count > 2048)

    @classmethod
    def tearDownClass(cls):
        if cls.process:
            kill_process_tree(cls.process.pid)
        cls.out_log_file.close()
        os.remove(cls.out_log_name)
        cls.err_log_file.close()
        os.remove(cls.err_log_name)


class TestNPULogRequestLevel1(CustomTestCase):
    log_requests_level = 1


class TestNPULogRequestLevel2(CustomTestCase):
    log_requests_level = 2


class TestNPULogRequestLevel3(CustomTestCase):
    log_requests_level = 3


if __name__ == "__main__":
    unittest.main()
