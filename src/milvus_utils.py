from pymilvus import MilvusClient, CollectionSchema, FieldSchema, DataType, WeightedRanker, AnnSearchRequest, RRFRanker, Collection
from pymilvus.model.sparse.bm25.tokenizers import build_default_analyzer
from pymilvus.model.sparse import BM25EmbeddingFunction
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from scipy.sparse import csr_matrix
from rank_bm25 import BM25Okapi
from collections import Counter
import nltk
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
import numpy as np
import re
from FlagEmbedding import BGEM3FlagModel
import nltk

def sparse_to_milvus_format(sparse_list):
	milvus_sparse = []
	for vec in sparse_list:
		tmp = {}
		vec_csr = vec.tocsr()
		indices = vec_csr.nonzero()[0].tolist()
		values = vec_csr.data.tolist()
		for i,v in zip(indices,values):
			tmp[i] = float(v)
		milvus_sparse.append(tmp)
	
	return milvus_sparse

def get_embed(texts, option=None):
	print("=== Embedding ===")
	bge_m3_ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")
	embeddings = bge_m3_ef(texts)
	if option=='sparse':
		print("=== SPARSE ===")
		sparse_vectors = [vec.tocsr() for vec in embeddings['sparse']]
		sparse_vectors = sparse_to_milvus_format(sparse_vectors)
		return sparse_vectors
	elif option == 'dense':
		#dense_vectors = embeddings['dense']
		dense_vectors = [vec.astype(float).tolist() for vec in embeddings['dense']]
		print("=== DENSE ===")
		return dense_vectors
	elif option == 'hybrid':
		sparse_vectors = [vec.tocsr() for vec in embeddings['sparse']]
		sparse_vectors = sparse_to_milvus_format(sparse_vectors)
		dense_vectors = embeddings['dense']
		return dense_vectors, sparse_vectors
	else:
		raise Exception(f"Option is required (sparse or dense or hybrid) but {option} is received")

class MilvusUtil:
	def __init__(self, collection_name: str, uri: str):
		self.collection = collection_name
		self.uri = uri
		self.client = None
		self.schema = None

	def connect_milvus_with_local(self, dim=None, fields=None):
		self.client = MilvusClient(uri=self.uri)
		if self.client.has_collection(collection_name=self.collection):
			print(f"=== Collection '{self.collection}' already exists. Skipping creation. ===")
			return
		schema = CollectionSchema(fields=fields)
		self.schema = CollectionSchema(fields,"")

		self.client.create_collection(
			collection_name=self.collection,
			#dimension=dim,
			consistency_level="Eventually",
			auto_id=True,
			overwrite=True,
			schema=schema,
		)
		print(f"=== Collection '{self.collection}' created successfully ===")

	def insert(self, data: list):
		if self.client is None:
			raise Exception("Client not initialized. Call connect_milvus_with_local first.")
		print(f"=== Inserting DATA ===")
		try:
			col = Collection(self.collection, self.schema)
			col.insert(data, progress_bar=True)
			print(f"=== Inserted Complete ===")
		except Exception as e:
			print(f"****** ERROR OCCURRED: {e}")
			raise 
#		self.client.flush(self.collection)
#		self.client.load_collection(self.collection)

	def create_index(self, field: str, metric: str, idx_type="IVF_FLAT", idx_name="vector_index",option=None):
		if self.client is None:
			raise Exception("Client not initialized.")
		print("=== Creating Index ===")
		index_params = MilvusClient.prepare_index_params()
		
		if option == "sparse":
			"""
			index type : SPARSE_INVERTED_INDEX
			params : DAAT_WAND or TAAT_NAIVE or DAAT_MAXSCORE
			"""
			print("=== Sparse Vector Index ===")
			index_params.add_index(
				field_name=field,
				metric_type=metric,
				index_type=idx_type, # SPARSE_INVERTED_INDEX
				index_name=idx_name,
				params={"inverted_index_algo": "DAAT_MAXSCORE",},
			)
				
		if option== "dense":
			print("=== Dense Vector Index ===")
			index_params.add_index(
				field_name=field,
				metric_type=metric,
				index_type=idx_type,
				index_name=idx_name,
				params={"nlist": 128,},
			)
		
		self.client.create_index(
			collection_name=self.collection,
			index_params=index_params,
			sync=True,
		)
		print("=== Create Index successfully ===")
		self.desc_index(idx_name)
		

	def delete_index(self, idx_name="vector_index"):
		if self.client is None:
			raise Exception("Client not initialized.")

		if self.client.list_indexes(collection_name=self.collection):
			self.client.drop_index(
				collection_name=self.collection,
				index_name=idx_name,
			)
			print("=== DELETE COMPLETE ===")
			self.desc_index()
		else:
			print("## NO INDEX EXISTS")
	
	def query(self, dense_query=None, sparse_query=None, 
			output_fields=None, 
			metric ="IP", 
			top_k=3,
			field_name="vector",
			method="dense"):

		if method == "sparse":
			print("=== SPARSE VECTOR SEARCH ===")
			search_kwargs = {
				"collection_name": self.collection,
				"data": sparse_query,
				"output_fields": output_fields,
				"anns_field" : field_name,
				"limit": top_k,
				#"consistency_level": "Eventually",
				#"search_params" : {"metric_type": metric}
			}
		elif method == "dense":
			print("=== DENSE VECTOR SEARCH ===")
			search_kwargs = {
				"collection_name": self.collection,
				"data": dense_query,
				"output_fields": output_fields,
				"anns_field" : field_name,
				"limit": top_k,
				"consistency_level": "Eventually",
				"search_params" : {"metric_type": metric}
			}
		elif method == 'hybrid':
			print("=== HYBRID SEARCH ===")
			search_kwargs_1 = {
				"data": dense_query,
				"anns_field" : "vector",
				"limit": top_k,
				"param" : {
					"metric_type" : metric,
				}
			}
			request_1 = AnnSearchRequest(**search_kwargs_1)
			search_kwargs_2 = {
				"data": sparse_query,
				"anns_field" : "sparse_vector",
				"limit": top_k,
				"param" : {
					"metric_type": metric
				}
			}
			request_2 = AnnSearchRequest(**search_kwargs_2)

			outputs = [request_1, request_2]
			ranker = WeightedRanker(0.5,0.5)
			res = self.client.hybrid_search(
				collection_name = self.collection,
				reqs = outputs,
				output_fields = output_fields,
				ranker = ranker,
				limit = top_k
			)
			return res
		else:
			print("[ERROR] REQUIRE SEARCH METHOD")
			raise Exception(f"REQUIRE SEARCH METHOD CURRENT METHOD IS {option} require(sparse, dense, hybrid)")

		result = self.client.search(**search_kwargs)
		return result
		

	def desc_collection(self):
		if self.client is None:
			raise Exception("Client not initialized.")
		print("=== DESC collection ===")
		print(self.client.describe_collection(collection_name=self.collection))
		print(self.client.get_collection_stats(self.collection))

	def desc_index(self, idx_name="vector_index"):
		if self.client is None:
			raise Exception("Client not initialized.")

		res = self.client.list_indexes(collection_name=self.collection)
		if res:
			print("=== List of INDEXES ===\n", res)
		else:
			print("# NO INDEX FOUND")
			return

		res = self.client.describe_index(
			collection_name=self.collection,
			index_name=idx_name
		)
		print("=== Described INDEX ===\n", res)


def main():
	fields = [
		FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
		FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=4),
		FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512)
	]

	milvus = MilvusUtil(collection_name="CBNU", uri="../db/test.db")
	milvus.connect_milvus_with_local(dim=4, fields=fields)
	milvus.insert([{"vector": [1,2,3,4], "title": "My Doc"}])
	milvus.create_index(field="vector", metric="L2")
	milvus.desc_index()
	milvus.desc_collection()

