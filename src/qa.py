import json
import numpy as np
import pandas as pd
import os
import sys
import click
import requests
from preprocess import Preprocess
from vllm import LLM, SamplingParams
from pprint import pp


result_buffer = []
prompt_dir="../prompt/rag_prompt.txt"
with open(prompt_dir,"r") as f:
	template = f.read()

@click.command()
@click.option('--data', default="../data/md_re_files",help="<path/to/data>")
@click.option('--meta', default="../data/rag_documents",help="<path/to/meta>")
@click.option('--output',default="../data/qa")
@click.option('--llm_server',default=0)
def main(data,meta,output,llm_server):
	global template
	host="http://localhost:8888/v1/completions"
	headers= {"Content-Type":"application/json"}
	pre = Preprocess(data,meta)
	metadata = pre.load_data_and_meta("md","json")
	docs = pre.docs
	result_path=os.path.join(output,"qa_deep.txt")
	result = open(result_path,"a")
	for i in docs:
		source_path = i.metadata.get("source","")
		base_name = os.path.basename(source_path)
		i.metadata = metadata.get(os.path.splitext(base_name)[0],{}).get("url","")

	if llm_server:
		data = {
			"prompt":prompt,
			"temperature":0,
			"max_tokens": 1024
		}
		response = requests.post(host, headers=headers, data=json.dumps(data))
		print(pp(response.json()))
	
	else:
		llm= LLM(model="../model/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/", tensor_parallel_size=2)
		sampling_params = SamplingParams(
			temperature=0.0,
			top_p=0.9,
			max_tokens=1024,
		)
		batch_size = 64
		for i in range(0,len(docs),batch_size):
			batch_docs = docs[i:i+batch_size]
			prompts = [template.format(context=doc.page_content) for  doc in batch_docs]
			
			outputs = llm.generate(prompts, sampling_params)

			for output in outputs:
				text = output.outputs[0].text
				result_buffer.append(text)
	result.write("\n".join(result_buffer))
	result.close()
	
if __name__=="__main__":
	main()
	
