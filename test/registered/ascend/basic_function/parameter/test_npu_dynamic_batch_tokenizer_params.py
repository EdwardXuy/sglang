
import threading
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import (
    LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH,
    send_concurrent_requests,
    verify_process_terminated,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=600, suite="nightly-1-npu-a3", nightly=True)

NUM_REQUESTS = 20
NUM_CONCURRENT = 8

SHORT_PROMPTS = ["Hi", "OK", "Yes"]
MEDIUM_PROMPTS = [
    "What is the capital of France?",
    "Explain what a neural network is.",
    "Describe the water cycle briefly.",
]
LONG_PROMPTS = [
    "Describe the history of the Roman Empire and its influence on modern culture " * 3,
    "Explain how large language models are trained, evaluated, and deployed " * 3,
]

SAMPLING_CONFIGS = [
    {"temperature": 0.0, "max_new_tokens": 32},
    {"temperature": 0.7, "max_new_tokens": 32},
    {"temperature": 1.0, "max_new_tokens": 32},
    {"temperature": 0.0, "top_p": 0.9, "max_new_tokens": 32},
]


# ------------------------------------------------------------------------------
# Batch size boundary tests
# ------------------------------------------------------------------------------

class TestDynamicBatchTokenizerBatchSize64(CustomTestCase):
    """Testcase: 80 requests with batch_size=64; tokenizer must process two
    successive cycles (64 + 16) without dropping any request.

    [Test Category] Parameter
    [Test Target] --enable-dynamic-batch-tokenizer; --dynamic-batch-tokenizer-batch-size
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    batch_size = 64
    num_requests = 80
    num_concurrent = 16

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--enable-dynamic-batch-tokenizer",
                "--dynamic-batch-tokenizer-batch-size",
                str(cls.batch_size),
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        verify_process_terminated(cls.process, cls.__name__)

    def test_batch_size_64(self):
        results = send_concurrent_requests(
            base_url=self.base_url,
            num_requests=self.num_requests,
            num_concurrent=self.num_concurrent,
        )
        success_count = sum(1 for r in results if r["status_code"] == 200)
        self.assertEqual(
            success_count,
            self.num_requests,
            f"Expected {self.num_requests} successes, got {success_count}.",
        )


class TestDynamicBatchTokenizerBatchSize1(CustomTestCase):
    """Testcase: batch_size=1 disables grouping; each request is tokenized
    individually but all must complete correctly.

    [Test Category] Parameter
    [Test Target] --enable-dynamic-batch-tokenizer; --dynamic-batch-tokenizer-batch-size
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    batch_size = 1

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--enable-dynamic-batch-tokenizer",
                "--dynamic-batch-tokenizer-batch-size",
                str(cls.batch_size),
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        verify_process_terminated(cls.process, cls.__name__)

    def test_batch_size_1(self):
        results = send_concurrent_requests(
            base_url=self.base_url,
            num_requests=NUM_REQUESTS,
            num_concurrent=4,
        )
        success_count = sum(1 for r in results if r["status_code"] == 200)
        self.assertEqual(
            success_count,
            NUM_REQUESTS,
            f"Expected {NUM_REQUESTS} successes, got {success_count}.",
        )
        for r in results:
            self.assertIn(
                "Paris",
                r["text"],
                f"Task {r['task_id']}: result missing 'Paris'.",
            )


# ------------------------------------------------------------------------------
# Timeout boundary tests (0.001 s and 0.1 s)
# ------------------------------------------------------------------------------

class TestDynamicBatchTokenizerTimeout001(CustomTestCase):
    """Testcase: batch_wait_timeout=0.001 s (near-zero); minimal accumulation
    window, effective batch_size in debug logs stays close to 1.

    [Test Category] Parameter
    [Test Target] --enable-dynamic-batch-tokenizer; --dynamic-batch-tokenizer-batch-timeout
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
    batch_timeout = 0.001

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--enable-dynamic-batch-tokenizer",
                "--dynamic-batch-tokenizer-batch-timeout",
                str(cls.batch_timeout),
                "--log-level",
                "debug",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        verify_process_terminated(cls.process, cls.__name__)

    def test_requests_succeed(self):
        results = send_concurrent_requests(
            base_url=self.base_url,
            num_requests=NUM_REQUESTS,
            num_concurrent=NUM_CONCURRENT,
        )
        success_count = sum(1 for r in results if r["status_code"] == 200)
        self.assertEqual(
            success_count,
            NUM_REQUESTS,
            f"[timeout={self.batch_timeout}] Expected {NUM_REQUESTS} successes, "
            f"got {success_count}.",
        )
        for r in results:
            self.assertIn(
                "Paris",
                r["text"],
                f"[timeout={self.batch_timeout}] Task {r['task_id']}: "
                f"result missing 'Paris'.",
            )


class TestDynamicBatchTokenizerTimeout01(TestDynamicBatchTokenizerTimeout001):
    """Testcase: batch_wait_timeout=0.1 s (50x default); long accumulation window
    maximises batch_size visible in debug logs; correctness must be unaffected.

    [Test Category] Parameter
    [Test Target] --enable-dynamic-batch-tokenizer; --dynamic-batch-tokenizer-batch-timeout
    """

    batch_timeout = 0.1


# ------------------------------------------------------------------------------
# Different sampling parameters (kwargs mismatch prevents batch grouping)
# ------------------------------------------------------------------------------

class TestDynamicBatchTokenizerSamplingParams(CustomTestCase):
    """Testcase: concurrent requests with different sampling kwargs are processed
    individually by the tokenizer; all 20 must return HTTP 200.

    [Test Category] Parameter
    [Test Target] --enable-dynamic-batch-tokenizer; --dynamic-batch-tokenizer-batch-size;
                  --dynamic-batch-tokenizer-batch-timeout
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--enable-dynamic-batch-tokenizer",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        verify_process_terminated(cls.process, cls.__name__)

    def test_different_sampling_params(self):
        payloads = SAMPLING_CONFIGS * 5  # 20 requests total
        results = []
        lock = threading.Lock()

        def _send(sampling_params):
            try:
                resp = requests.post(
                    f"{self.base_url}/generate",
                    json={
                        "text": "The capital of France is",
                        "sampling_params": sampling_params,
                    },
                    timeout=60,
                )
                with lock:
                    results.append({"status_code": resp.status_code})
            except Exception as exc:
                with lock:
                    results.append({"status_code": -1, "text": str(exc)})

        threads = [threading.Thread(target=_send, args=(p,)) for p in payloads]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        success_count = sum(1 for r in results if r["status_code"] == 200)
        self.assertEqual(
            success_count,
            len(payloads),
            f"Expected {len(payloads)} successes, got {success_count}.",
        )


# ------------------------------------------------------------------------------
# Combo: high concurrency + mixed text lengths + streaming (shared server)
# ------------------------------------------------------------------------------

class TestDynamicBatchTokenizerCombo(CustomTestCase):
    """Testcase: combined parameters covering high concurrency, mixed prompt
    lengths, and SSE streaming.  All three test methods share one server
    launched with batch_size=16, batch_timeout=0.01, --disable-radix-cache.

    [Test Category] Parameter
    [Test Target] --enable-dynamic-batch-tokenizer; --dynamic-batch-tokenizer-batch-size;
                  --dynamic-batch-tokenizer-batch-timeout; --disable-radix-cache
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "ascend",
                "--disable-cuda-graph",
                "--enable-dynamic-batch-tokenizer",
                "--dynamic-batch-tokenizer-batch-size",
                "16",
                "--dynamic-batch-tokenizer-batch-timeout",
                "0.01",
                "--disable-radix-cache",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)
        verify_process_terminated(cls.process, cls.__name__)

    def test_high_concurrency(self):
        # 100 requests with batch_size=16: ~7 tokenization cycles are formed.
        results = send_concurrent_requests(
            base_url=self.base_url,
            num_requests=100,
            num_concurrent=20,
        )
        success_count = sum(1 for r in results if r["status_code"] == 200)
        self.assertEqual(
            success_count,
            100,
            f"Expected 100 successes, got {success_count}.",
        )
        for r in results:
            self.assertIn("Paris", r["text"])

    def test_mixed_text_lengths(self):
        # Short / medium / long prompts may land in the same accumulation window.
        all_prompts = SHORT_PROMPTS + MEDIUM_PROMPTS + LONG_PROMPTS
        results = []
        lock = threading.Lock()

        def _send(prompt):
            try:
                resp = requests.post(
                    f"{self.base_url}/generate",
                    json={
                        "text": prompt,
                        "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                    },
                    timeout=60,
                )
                with lock:
                    results.append({"status_code": resp.status_code})
            except Exception as exc:
                with lock:
                    results.append({"status_code": -1, "text": str(exc)})

        threads = [threading.Thread(target=_send, args=(p,)) for p in all_prompts]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        success_count = sum(1 for r in results if r["status_code"] == 200)
        self.assertEqual(
            success_count,
            len(all_prompts),
            f"Expected {len(all_prompts)} successes, got {success_count}.",
        )

    def test_streaming_requests(self):
        # "stream": True is a top-level field; requests.post must also use stream=True.
        results = []
        lock = threading.Lock()

        def _send_stream(prompt):
            try:
                resp = requests.post(
                    f"{self.base_url}/generate",
                    json={
                        "text": prompt,
                        "sampling_params": {"temperature": 0, "max_new_tokens": 32},
                        "stream": True,
                    },
                    stream=True,
                    timeout=60,
                )
                has_content = any(
                    line
                    and line.startswith(b"data: ")
                    and line[6:] != b"[DONE]"
                    for line in resp.iter_lines()
                )
                with lock:
                    results.append(
                        {"status_code": resp.status_code, "has_content": has_content}
                    )
            except Exception as exc:
                with lock:
                    results.append({"status_code": -1, "has_content": False})

        prompts = [
            "The capital of France is",
            "The largest planet in the solar system is",
            "The speed of light is approximately",
        ]
        threads = [threading.Thread(target=_send_stream, args=(p,)) for p in prompts]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for r in results:
            self.assertEqual(
                r["status_code"], 200, "Streaming request returned non-200 status."
            )
            self.assertTrue(
                r["has_content"], "Streaming response returned no SSE content chunks."
            )


if __name__ == "__main__":
    unittest.main()