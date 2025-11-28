#!/bin/bash

set -x

# 更新vllm-ascend调试代码
bash /home/liudi/vllm-ascend/my_test/upload.sh


export VLLM_LOGGING_LEVEL=INFO
# arg_utils.py
export VLLM_USE_V1=1
#export VLLM_WORKER_MULTIPROC_METHOD=forkserver

# export ASCEND_RT_VISIBLE_DEVICES=0,1

cd /home/ml/weight/

python -m vllm.entrypoints.openai.api_server \
       --model="Qwen3-8B-W8A8" \
       --served-model-name qwen3_moe \
       --gpu-memory-utilization 0.9 \
       --max-num-seqs 768 \
       --max-model-len 22528 \
       --trust-remote-code \
       --enforce-eager \
       --distributed_executor_backend=mp \
       --tensor-parallel-size 2 \
       --port 8000 \
		--enforce-eager \
		--compilation-config '{"cudagraph_capture_sizes": [1]}' \
		--scheduling-policy "sjf" \
		--min-prefill-batch-size 2 \
		--prefill-request-batching-timeout-ms 10000 \
		--scheduler-delay-us 1000000


cd -



