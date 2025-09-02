#!/bin/bash
MODEL="binwon/kanana-cot"

MODEL="../model/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/"
python -m vllm.entrypoints.openai.api_server --model $MODEL --gpu-memory-utilization 0.2 --port 8888 --tensor-parallel-size 2
