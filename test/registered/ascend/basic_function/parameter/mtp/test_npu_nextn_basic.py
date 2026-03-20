

import os
import unittest



from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH,
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
    "--disable-radix-cache",
    # Use NEXTN algorithm (MTP) – no draft model needed
    "--speculative-algorithm", "NEXTN", #or EAGLE
    # Number of auto-regressive steps per iteration (tune based on GPU memory)
    "--speculative-num-steps", "2",   # 3 for EAGLE: Lower for memory, increase for speed
    # Branching factor (1 = greedy, >1 for speculative sampling, SPEC-V2 now only support 1)
    "--speculative-eagle-topk", "1",
    # Maximum draft tokens to verify per step
    "--speculative-num-draft-tokens", "3", # 5 for EAGLE
    "--speculative-attention-mode", "decode",
    # Tensor parallelism – adjust according to available NPUs
    "--tp-size", "16",                # 16 for large model with 8+ NPUs
    "--mem-fraction-static", "0.9", # 0.85 for EAGLE
    "--disable-cuda-graph",
    "--dtype", "bfloat16",
]


class TestNpuNextnBasic(CustomTestCase):
    """
    [Test Category] Parameter
    [Test Target] --speculative-algorithm; --speculative-num-steps;
                  --speculative-eagle-topk; --speculative-num-draft-tokens;
    [Model] DeepSeek-V3.2-W8A8 (vllm-ascend/DeepSeek-V3.2-W8A8)
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.base_url = DEFAULT_URL_FOR_TEST
        os.environ.update({
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
            "SGLANG_ENABLE_SPEC_V2": "1",
        })
        cls.process = popen_launch_server(
            DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=_SERVER_ARGS,
            env=os.environ.copy(),
        )

    @classmethod
    def tearDownClass(cls) -> None:
        kill_process_tree(cls.process.pid)

    def test_nextn_basic_inference(self):
        """
        Test steps:
          1. Send a single inference request to the NEXTN-enabled server.
          2. Assert the response payload is structurally valid.
          3. Assert avg_spec_accept_length > 1.0 (multi-token acceptance confirmed).
        """
        response = send_inference_request(
            self.base_url, DEEPSEEK_V3_2_W8A8_WEIGHTS_PATH,
            "List 3 programming languages and their primary use cases.",
        )

        content = response["choices"][0]["message"]["content"]
        print(f"Q: List 3 programming languages and their primary use cases")
        print(f"A; {content}")

        self.assertIn("choices", response)
        self.assertGreater(len(response["choices"]), 0)
        self.assertGreater(
            len(response["choices"][0]["message"]["content"].strip()), 0
        )

        assert_spec_decoding_active(self, self.base_url, threshold=1.0)


if __name__ == "__main__":
    unittest.main()
