from huggingface_hub import snapshot_download
import sys

if len(sys.argv) !=2:
	print("* model name is required")
	print("* Usage: ./model_download.py <huggingface_model_name>")
model_name=sys.argv[1]

snapshot_download(
	repo_id=model_name,
	local_dir="/data/jks/vllm/model/"+model_name
)
