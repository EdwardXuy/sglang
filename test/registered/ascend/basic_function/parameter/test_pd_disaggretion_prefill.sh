export ASCEND_MF_STORE_URL="tcp://141.61.29.204:8000"

python -m sglang.launch_server \
    --model-path /home/weights/Qwen3-VL-30B-A3B-Instruct \
    --disaggregation-mode prefill \
    --host 141.61.29.204 \
    --port 8080 \
    --trust-remote-code \
    --base-gpu-id 14 \
    --tp-size 2 \
    --mem-fraction-static 0.9 \
    --attention-backend ascend \
    --device npu \
    --disaggregation-transfer-backend ascend \
    --attention-backend ascend \
    --log-level debug \
    --log-level-http debug \
    --disaggregation-bootstrap-port 8998 \
    --dtype bfloat16 
