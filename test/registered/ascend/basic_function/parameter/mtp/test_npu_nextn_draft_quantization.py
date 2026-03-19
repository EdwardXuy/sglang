
import os
import unittest



from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    QWEN3_0_6B_WEIGHTS_PATH,
    QWEN3_30B_A3B_W8A8_WEIGHTS_PATH,
    assert_spec_decoding_active,
    send_inference_request,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

_ASCEND_BACKEND = "ascend"

_SERVER_ARGS = [
    "--trust-remote-code",
    "--attention-backend", _ASCEND_BACKEND,
    "--quantization", "modelslim",
    "--disable-radix-cache",
    "--speculative-algorithm", "NEXTN",
    "--speculative-draft-model-path", QWEN3_0_6B_WEIGHTS_PATH,
    "--speculative-num-steps", "4",
    "--speculative-eagle-topk", "1",
    "--speculative-num-draft-tokens", "7",
    # --speculative-draft-model-quantization unquant: load draft model in full
    # precision regardless of the target model quantization scheme.
    # Allowed values: unquant, modelslim, or any other backend-supported scheme.
    "--speculative-draft-model-quantization", "unquant",
    "--speculative-attention-mode", "decode",
    "--tp-size", "1",
    "--mem-fraction-static", "0.7",
    "--disable-cuda-graph",
    "--dtype", "bfloat16",
]


class TestNpuNextnDraftQuantization(CustomTestCase):
    """
    [Test Category] Parameter
    [Test Target] --speculative-draft-model-quantization; --speculative-algorithm;
                  --speculative-num-steps; --speculative-num-draft-tokens;
                  --speculative-eagle-topk;
    [Model] Qwen/Qwen3-30B-A3B-w8a8; Qwen/Qwen3-0.6B
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            QWEN3_30B_A3B_W8A8_WEIGHTS_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=_SERVER_ARGS,
            env=os.environ.copy(),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        kill_process_tree(cls.process.pid)

    def test_nextn_draft_quantization(self):
        """
        Test steps:
          1. Send a single inference request with the mixed-precision configuration.
          2. Assert the response payload is structurally valid.
          3. Assert avg_spec_accept_length > 1.0 to confirm the unquantized draft
             model cooperates correctly with the quantized target model.
        """
        response = send_inference_request(
            self.base_url, QWEN3_30B_A3B_W8A8_WEIGHTS_PATH,
            "What is the Python programming language?",
        )

        content = response["choices"][0]["message"]["content"]
        print(f"Q: What is the Python programming language")
        print(f"A: {content}")

        self.assertIn("choices", response)
        self.assertGreater(len(response["choices"]), 0)
        self.assertGreater(
            len(response["choices"][0]["message"]["content"].strip()), 0
        )

        # avg_spec_accept_length > 1.0: if --speculative-draft-model-quantization
        # were silently ignored, logit distribution mismatch would collapse
        # acceptance to near zero without raising any exception.
        assert_spec_decoding_active(self, self.base_url, threshold=1.0)


if __name__ == "__main__":
    unittest.main()
