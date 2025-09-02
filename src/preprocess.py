import requests
import json
import glob
import numpy as np
from boilerpy3 import extractors
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
import os
from vllm import LLM


def get_vllm_embeddings_batched(texts, model, endpoint, batch_size=1024):
	headers = {"Content-Type": "application/json"}
	all_embeddings = []

	for i in range(0, len(texts), batch_size):
		batch = texts[i:i + batch_size]
		data = {
			"input": batch,
			"model": model
		}

		response = requests.post(endpoint, headers=headers, data=json.dumps(data))
		response.raise_for_status()

		embeddings = [item["embedding"] for item in response.json()["data"]]
		all_embeddings.extend(embeddings)

		print(f"=== Embedded batch {i} to {i + len(batch)} ===")
	return all_embeddings

def get_embed_from_model(texts,model,batch_size=1024):
	embeddings = []
	model = LLM(model=model, 
				task="embed",
				enforce_eager=True
	)
	for i in range(0, len(texts), batch_size):
		batch = texts[i:i + batch_size]
		outputs = model.embed(batch)
		for output in outputs:
			embeddings.append(output.outputs.embedding)
	return embeddings

class Preprocess:
	def __init__(self, data_dir, meta_dir):
		self.meta_path = meta_dir
		self.data_path = data_dir
		self.extractor = extractors.ArticleExtractor()
		self.docs = []

	def load_data_and_meta(self,data_glob, meta_glob):
		metadata_map = {}
		meta_glob = "/*."+meta_glob
		glob_path = self.meta_path + meta_glob

		data_glob = "*." + data_glob

		for path in glob.glob(glob_path):
			with open(path, "r", encoding="utf-8") as f:
				meta = json.load(f)
				if isinstance(meta, list):
					meta = meta[0]
				metadata_map[meta["id"]] = meta

		loader = DirectoryLoader(
			path=self.data_path,
			glob=data_glob,
			show_progress=True,
			silent_errors=True
		)
		self.docs = loader.load()
		return metadata_map

	def refine_docs(self):
		refine_docs = []
		for doc in self.docs:
			source_path = doc.metadata.get('source')
			try:
				content = self.extractor.get_content_from_file(source_path)
				refine_doc = Document(page_content = content, metadata = {"source":source_path})
				refine_docs.append(refine_doc)
			except Exception as e:
				print(f"[WARN] Failed to extract text from {source_path}: {e}")
		self.docs = refine_docs

	def split_data(self, model_name=None, chunk_size=128, option="None"):
		CHUNK_OVERLAP = int(chunk_size * 0.2)
		if option != "semantic" and option != "recursive":
			raise Exception(f"[ERROR] Option is <required> semantic or recursive. but, receive {option}")

		if option=="recursive":
			print(f"=== Recursive Character Text Splitter ===")
			splitter = RecursiveCharacterTextSplitter(
				chunk_size=chunk_size,
				chunk_overlap=CHUNK_OVERLAP
			)
			chunks = splitter.split_documents(self.docs)
		elif option == "semantic":
			print(f"=== Semantic Text Splitter ===")
			print(f"=== MODEL : {model_name} ===")
			embedding_model = HuggingFaceEmbeddings(model_name=model_name)
			splitter = SemanticChunker(embedding_model)
			chunks = splitter.split_documents(self.docs)

		print(f"=== Split into {len(chunks)} chunks. ===")

		texts = [doc.page_content for doc in chunks]
		return chunks, texts

	def get_embed_data(self, texts, model_path, model_name="None"):
		print(f"=== Sending texts for embedding ... ===")
		if model_name !="None":
			embeddings = get_vllm_embeddings_batched(texts, model_name, model_path)
		elif model_name == "None":
			embeddings = get_embed_from_model(texts,model_path)
		print(f"=== Finish Embedding successfully ===")
		embeddings = np.array(embeddings)
		embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

		return embeddings

def main():
	data_path = "../data"
	meta_path = "../data"

	model_endpoint = "http://localhost:8000/v1"
	model_name = "Qwen3-embedding"

	pre = Preprocess(data=data_path, meta=meta_path)
	metadata_map = pre.load_data_and_meta()
	refine_texts = pre.text_refine()
	chunks, texts = pre.split_data()
	embeddings = pre.embed_data(texts, model_endpoint, model_name)

	print("✅ All preprocessing done.")
