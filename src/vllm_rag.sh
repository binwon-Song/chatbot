#!/bin/bash

#CHAT_MODEL="binwon/kanana-cot"
CHAT_MODEL="kakaocorp/kanana-nano-2.1b-instruct"
EMBED_MODEL="../model/bge-korean"
DB="../db/hybrid.db"
CHAT_PORT=8888
check_port=$(lsof -Pi :$CHAT_PORT -sTCP:LISTEN -t)

if [ -n "$check_port" ]; then
	echo "# VLLM $CHAT_MODEL is running"
else
	echo "# Not running vllm chat server"
	echo "# Starting VLLM Server $CHAT_MODEL"
	vllm serve $CHAT_MODEL --gpu-memory-utilization 0.3 --max-model-len 2048 --port $CHAT_PORT
	exit 1
fi

COMMAND="python rag.py --chat-model $CHAT_MODEL "
COMMAND+="--vllm-chat-endpoint http://127.0.0.1:$CHAT_PORT/v1 "
COMMAND+="--embedding-model $EMBED_MODEL --uri $DB -i"

$COMMAND
