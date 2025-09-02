import json
import uuid
import numpy as np
import time
from pymilvus import FieldSchema, DataType, connections
import os
import sys
from milvus_utils import MilvusUtil
import milvus_utils as mu
from preprocess import Preprocess
from sentence_transformers import SentenceTransformer
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
import pickle

if len(sys.argv) !=2:
	print("Require arguments </path/to/db> ")
	sys.exit()

MILVUS_URI=sys.argv[1]

COLLECTION_NAME = "CBNU"

# VLLM_EMBEDDING_ENDPOINT = "http://localhost:8000/v1/embeddings"

DATA_PATH = "../data/md_re_files/"
META_PATH = "../data/rag_documents/"

#DATA_PATH = "./"
#META_PATH = "./"

### create instance of vector db and preprocessing ###
milvus = MilvusUtil(collection_name = COLLECTION_NAME, uri=MILVUS_URI)
pre = Preprocess(DATA_PATH, META_PATH)

### load data and metadata ###
metadata = pre.load_data_and_meta(data_glob = "md", meta_glob = "json")

### refinement text of document ###
### Get main content of html ###
#pre.refine_docs()

### split data chunking ###
### get text from chunked docs ###
CHUNK_SIZE = 256
chunks_docs, texts = pre.split_data(option="recursive", chunk_size=CHUNK_SIZE)

### embedding texts from model ###
#embeddings = pre.get_embed_data(texts, MODEL_PATH)
dense_vectors, sparse_vectors = mu.get_embed(texts,"hybrid")
#embeddings = mu.get_embed(texts)
#dense_vectors = embeddings['dense']
#sparse_vectors, bm25 = milvus.ko_sparse_embed(docs = texts, option="insert")
#sparse_vectors = [vec.tocsr() for vec in embeddings['sparse']]
#sparse_vectors = mu.sparse_to_milvus_format(sparse_vectors)

#with open("bm25.pkl","wb") as f:
#	pickle.dump(bm25,f)

dim=dense_vectors[0].shape[0]
### vector db columns ###
fields = [
	FieldSchema(name="id", dtype=DataType.INT64, auto_id=True, is_primary=True),
	FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
	FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=1024),
	FieldSchema(name="text", dtype=DataType.VARCHAR, description="Chunk content", max_length=10240),
	FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, description="Embedding vector", dim=dim),
	FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR, description="Sparse vector"),
]

### connecting vector db and create collection ###
connections.connect("default", uri=MILVUS_URI)
milvus.connect_milvus_with_local(dim=dim,fields=fields)

### Milvus input data ###
dict_list = []
for chunk, vector, sparse in zip(chunks_docs, dense_vectors, sparse_vectors):
	source_path= chunk.metadata.get("source","")
	base_name= os.path.basename(source_path)
	doc_id=os.path.splitext(base_name)[0]
	meta = metadata.get(doc_id,{})
	chunk_dict = {
		"text": chunk.page_content,
		"vector": vector.astype(np.float32),
		"sparse_vector": sparse,
		"title": meta.get("title",""),
		"source": meta.get("url",""),
	}
	dict_list.append(chunk_dict)

### Milvus 삽입 ###
BATCH_SIZE=100
print("📦 Inserting into Milvus...")
print(f"dict size: {len(dict_list)} , {len(dict_list)/BATCH_SIZE} insert required")
start_time = time.time()

for i in range(0,len(dict_list),BATCH_SIZE):
	batch=dict_list[i:i+BATCH_SIZE]
	milvus.insert(batch)
	print(f"=== Inserted {min(i + BATCH_SIZE, len(dict_list))}. ===")

milvus.client.flush(COLLECTION_NAME)

end_time = time.time()
print(f"=== Inserted {len(dict_list)} vectors in {round(end_time - start_time, 2)} seconds. ===")
