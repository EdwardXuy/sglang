#!/bin/bash

pkill -9 sglang
pkill -9 python

echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
sysctl -w vm.swappiness=0
sysctl -w kernel.numa_balancing=0
sysctl -w kernel.sched_migration_cost_ns=50000

export SGLANG_SET_CPU_AFFINITY=1
unset https_proxy
unset http_proxy
unset HTTPS_PROXY
unset HTTP_PROXY
unset ASCEND_LAUNCH_BLOCKING

. /usr/local/Ascend/cann/set_env.sh
. /usr/local/Ascend/nnal/atb/set_env.sh

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export STREAMS_PER_DEVICE=32

MODEL_PATH="/root/.cache/modelscope/hub/models/DeepSeek-V3.2-W8A8"

P_IP=('your prefill ip1' 'your prefill ip2')
P_IFNAMES=('xxx' 'xxx')
D_IP=('your decode ip1' 'your decode ip2')
D_IFNAMES=('xxx' 'xxx')

export ASCEND_MF_STORE_URL=tcp://${P_IP[0]}:24667

# get IP in current node
LOCAL_HOST1=`hostname -I|awk -F " " '{print$1}'`
LOCAL_HOST2=`hostname -I|awk -F " " '{print$2}'`
echo "LOCAL_HOST1 = " ${LOCAL_HOST1}
echo "LOCAL_HOST2 = " ${LOCAL_HOST2}

# prefill
for i in "${!P_IP[@]}";
do
  if [[ "$LOCAL_HOST1" == "${P_IP[$i]}" || "$LOCAL_HOST2" == "${P_IP[$i]}" ]]; then
    mkdir -p log
    PREFILL_LOG_FILE="./log/launch_prefill_$(date +'%Y-%m-%d-%H:%M').log"

    echo "Node Rank : ${i}"
    NODE_RANK=$i
    export HCCL_BUFFSIZE=1200
    export DEEP_NORMAL_MODE_USE_INT8_QUANT=1
    export TASK_QUEUE_ENABLE=2
    export HCCL_SOCKET_IFNAME=${P_IFNAMES[$NODE_RANK]}
    export GLOO_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME}
    echo "GLOO_SOCKET_IFNAME : ${GLOO_SOCKET_IFNAME}"

    nnodes=${#P_IP[@]}
    tp_size=`expr 16 \* ${nnodes}`

    nohup python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
      --tp $tp_size \
      --trust-remote-code \
      --attention-backend ascend \
      --device npu \
      --watchdog-timeout 9000 \
      --host ${P_IP[$NODE_RANK]} --port 8000 \
      --mem-fraction-static 0.73 \
      --chunked-prefill-size -1 --max-prefill-tokens 68000 \
      --max-running-requests 1 \
      --moe-a2a-backend deepep --deepep-mode normal \
      --quantization modelslim \
      --disaggregation-transfer-backend ascend \
      --disaggregation-mode prefill \
      --disable-cuda-graph \
      --nnodes $nnodes --node-rank $NODE_RANK \
      --disaggregation-bootstrap-port 8995 \
      --moe-dense-tp-size 1 \
      --enable-nsa-prefill-context-parallel \
      --nsa-prefill-cp-mode in-seq-split \
      --attn-cp-size 32 \
      --speculative-algorithm NEXTN --speculative-num-steps 1 --speculative-eagle-topk 1 --speculative-num-draft-tokens 2 \
      --dist-init-addr ${P_IP[0]}:10000 \
      > $PREFILL_LOG_FILE 2>&1 &
  fi
done

# decode
for i in "${!D_IP[@]}";
do
  if [[ "$LOCAL_HOST1" == "${D_IP[$i]}" || "$LOCAL_HOST2" == "${D_IP[$i]}" ]]; then
    mkdir -p log
    DECODE_LOG_FILE="./log/launch_decode_$(date +'%Y-%m-%d-%H:%M').log"

    echo "Node Rank : ${i}"
    NODE_RANK=$i
    export SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1
    export SGLANG_ENABLE_SPEC_V2=1
    export TASK_QUEUE_ENABLE=0
    export SGLANG_SCHEDULER_SKIP_ALL_GATHER=1
    export HCCL_BUFFSIZE=400
    export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=8
    export HCCL_SOCKET_IFNAME=${D_IFNAMES[$NODE_RANK]}
    export GLOO_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME}
    echo "GLOO_SOCKET_IFNAME : ${GLOO_SOCKET_IFNAME}"

    nnodes=${#D_IP[@]}
    tp_size=`expr 16 \* ${nnodes}`
    dp_size=8
    ep_size=$tp_size

    nohup python3 -m sglang.launch_server --model-path ${MODEL_PATH} \
      --tp $tp_size \
      --dp $dp_size \
      --ep $ep_size \
      --moe-dense-tp-size 1 \
      --enable-dp-attention \
      --enable-dp-lm-head \
      --trust-remote-code \
      --attention-backend ascend \
      --device npu \
      --watchdog-timeout 9000 \
      --host ${D_IP[$NODE_RANK]} --port 8001 \
      --mem-fraction-static 0.79 \
      --disable-radix-cache \
      --chunked-prefill-size -1 --max-prefill-tokens 68000 \
      --max-running-requests 32 \
      --cuda-graph-max-bs 4 \
      --moe-a2a-backend deepep \
      --deepep-mode low_latency \
      --quantization modelslim \
      --speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
      --disaggregation-transfer-backend ascend \
      --disaggregation-mode decode \
      --nnodes $nnodes --node-rank $NODE_RANK \
      --dist-init-addr ${D_IP[0]}:10000 \
      > $DECODE_LOG_FILE 2>&1 &
  fi
done

pause

# router
if [[ "$LOCAL_HOST1" == "${P_IP[0]}" || "$LOCAL_HOST2" == "${P_IP[0]}" ]]; then
    ROUTER_LOG_FILE="./log/launch_router_$(date +'%Y-%m-%d-%H:%M').log"
    nohup python -u -m sglang_router.launch_router \
      --pd-disaggregation \
      --policy cache_aware \
      --host 0.0.0.0 \
      --port 6688 \
      --prefill http://${P_IP[0]}:8000 8995\
      --decode http://${D_IP[0]}:8001 \
      --mini-lb \
    > $ROUTER_LOG_FILE 2>&1 &
fi
