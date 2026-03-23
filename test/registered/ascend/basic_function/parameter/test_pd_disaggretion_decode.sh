echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0r
sysctl -w kernel.numa balancing=0ec
sysctl -w kernel.sched_migration_cost_ns=50000
# bind cpu
export SGLANG SET CPU AFFINITY=1 
unset https proxy
unset http proxy
unset HTTPS PROXY
unset HTTP PROXY
export ASCEND LAUNCH BLOCKING=1 
# cann
source /usr/local/Ascend/ascend-toolkit/set env.sh
source /usr/local/Ascend/nnal/atb/set env.sh
export SGLANG DISAGGREGATION WAITING_TIMEOUT=3600
export STREAMS PER DEVICE=32
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=64
export HCCL BUFFSIZE=2500
export HCCL_OP_EXPANSION MODE=AIV
export HCCL SOCKET IFNAME=enp196sOf0
export GL0O SOCKET IFNAME=enp196s0f0
export SGLANG NPU PROFILING-0
#export SGLANG NPU PROFILING STAGE
export DEEPEP NORMAL_LONG SEQ ROUND=32
export DEEPEP NORMAL LONG SEQ PER ROUND TOKENS-4096
export ASCEND_MF_STORE_URL="tcp:/761.47.19.75:24669"
export SGLANG DISAGGREGATION BOOTSTRAP TIMEOUT=3600
export SGLANG_DISAGGREGATION_WAITING_TIMEOUT=3600
export SGLANG ENABLE SPEC V2=1
export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
python3 -m sglang.launch servera
--model-path /home/weights/Qwen3.5-397B-A17B-w8a8
--attention-backend ascend
--device npu \
--tp-size 16 --nnodes 1 --node rank 0 \
_-chunked-prefill-size -1 --max-prefill-tokens 65536 \
--disable radix-cache
--trust-remote-code l
--host 0.0.0.0 --max-running requests 256 \
--moe-a2a-backend deepep
--deepep-mode low latency
--mem-fraction-static 0.85
--port 8000 \
--cuda-graph-bs 1 2 3 4 8 9 10 11 12 13 14 15 16 \
--quantization modelslim \
--enable-multimodal
--mm-attention-backend ascend attn --max-total-tokens 1200000 \
--dtype bfloat16 --mamba-ssm-dtype bfloat16 --disaggregation-mode decode --disaggregation-transfer-backend ascend
--speculative-draft-model-quantization unquant --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
