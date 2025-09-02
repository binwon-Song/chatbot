import sys
import numpy as np
from milvus_utils import MilvusUtil
import milvus_utils as mu
from preprocess import Preprocess
from pymilvus import FieldSchema, DataType
import pickle

if len(sys.argv) != 2:
	print("* Require arguments </path/to/db> ")
	print("* Usage e.g. ./query ../db/md.db ")
	sys.exit()


uri=sys.argv[1]
query = input("Query : ")
query= [query]
pre =  Preprocess("","")

mc = MilvusUtil("CBNU",uri)

## load pretrained bm25
#with open("bm25.pkl","rb") as f:
#	bm25 = pickle.load(f)

#query_sparse_embed, _ = mc.ko_sparse_embed(docs = query, bm25 = bm25, option = "query")
query_dense_embed, query_sparse_embed = mu.get_embed(query, option="hybrid")
#query_sparse_embed = embeddings['sparse']
#query_sparse_embed = mu.sparse_to_milvus_format(query_sparse_embed)
#query_dense_embed = embeddings['dense']
#query_dense_embed = pre.get_embed_data(query,model_path)

#mc.connect_milvus_with_local(dim, fields)

mc.connect_milvus_with_local()
print("=== Description of collection ===")
mc.desc_collection()

l = mc.client.list_indexes(collection_name = mc.collection)
for i in l:
	mc.delete_index(i)

mc.create_index(field="sparse_vector", idx_type="SPARSE_INVERTED_INDEX",metric="IP",idx_name="sparse_index",option="sparse")
mc.create_index(field="vector", idx_type="IVF_FLAT",metric="IP",idx_name="dense_index",option="dense")

output = ["text", "title", "source"]
query_options={
	"dense_query" :query_dense_embed,
	"sparse_query" : query_sparse_embed,
	"output_fields" : output ,
	"top_k" : 5,
	"metric" : "IP",
	#"field_name" : "sparse_vector",
	"method" : "hybrid",
}
results = mc.query(**query_options)

result_vector=[]
print(f"Query : {query}")
for hits in results:
	print("TopK results:")
	for hit in hits:
		print(hit)
		print("ID:", hit["id"])
		print("Distance:", round(hit["distance"], 5)*100)
		entity = hit.get("entity", {})
		print("Text: ",entity.get("text"))
		print("Source: ",entity.get("source"))
		print("Title: ",entity.get("title"))
		result_vector.append(entity.get("vector"))
		print("-" * 40)

## remove zombie
import torch.distributed as dist

if dist.is_initialized():
    dist.destroy_process_group()

		
#for hit in results[0]:
#	print("ID:", hit["id"])
#	print("Distance:", round(hit["distance"], 3))
#	entity = hit.get("entity", {})
#	print("Text: ",entity.get("text"))
#	print("Source: ",entity.get("source"))
#	print("Title: ",entity.get("title"))
#	result_vector.append(entity.get("vector"))
#	print("-" * 40)
