import argparse
from argparse import Namespace
from typing import Any

from langchain_community.document_loaders import DirectoryLoader
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate,ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_milvus import Milvus
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient
from preprocess import Preprocess
import milvus_utils as mu

class KrEmbed():
	def __init__(self,model_path):
		self.pre = Preprocess("","")
		self.model = model_path
	def embed_query(self,texts):
		embeddings = self.pre.get_embed_data(texts,self.model)
		return embeddings.astype(float).tolist()[0]
		
class KrSparseEmbed():
	def embed_query(self,texts):
		return mu.get_embed([texts], option='sparse')[0]
		

class KrDenseEmbed():
	def embed_query(self,texts):
		return mu.get_embed([texts], option='dense')[0]

def init_vectorstore(config: dict[str, Any]):
	"""
	Initialize vector store with documents
	"""
	embeddings = KrDenseEmbed()
	#embeddings = KrEmbed(config["embedding_model"])
	return Milvus(
		collection_name="CBNU",
		embedding_function=embeddings,
		vector_field = "vector",
		connection_args={"uri": config["uri"]},
		text_field="text"
	)
def init_milvus(config):
	dense_embed = KrDenseEmbed()
	sparse_embed = KrSparseEmbed()
	dense_index_param = {
		"metric_type": "IP",
		"index_type": "IVF_FLAT"
	}
	sparse_index_param = {
		"meric_type" : "IP",
		"index_type": "SPARSE_INVERTED_INDEX"
	}
	return Milvus(
		collection_name="CBNU",
		embedding_function=[dense_embed, sparse_embed],
		index_params=[dense_index_param, sparse_index_param],
		vector_field = ["vector","sparse_vector"],
		connection_args={"uri": config["uri"]},
		text_field="text"
	)

def init_llm(config: dict[str, Any]):
	"""
	Initialize llm
	"""
	return ChatOpenAI(
		model=config["chat_model"],
		openai_api_key=config["vllm_api_key"],
		openai_api_base=config["vllm_chat_endpoint"],
	)


def get_qa_prompt_old():
	"""
	Get question answering prompt template
	"""
	template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.
Include the source(s) of your answer at the end in the format: [Source: URL or document name].
Question: {question}
Context: {context}

Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
<source>
... 
</source>
"""
	return PromptTemplate.from_template(template)

def get_qa_prompt():
    SYSTEM_PROMPT = """You are a helpful AI assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.
Respond in this format:
<source>
...
</source>
<reasoning>
...
</reasoning>
<answer>
...
</answer>
    """

    template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", """Question: {question}
Context: {context}"""),
    ])
    return template


def get_qr_prompt():
	"""
	Get Question rewriting prompt template
	"""
	template = """You are an software department assistant at chungbuk national university.
Given a question and its context, decontextualize the question by addressing coreference and omission issues. The resulting question should retain its original meaning and be as informative as possible, and should not duplicate any previously asked questions in the context.
Context:
Q : Which is the software deparment?
A : Software department is located in S4-1 in chungbuk university

Q : How many graduation grades are?
A : In software major, graduation grades split into 3 category, cultur class 30 grades and major class 110 grades. total is 140 grades needed.

Question: {question}
Rewrite: 
"""
	return PromptTemplate.from_template(template)

def format_docs(docs: list[Document]):
	"""
	Format documents for prompt
	"""
	formatted=[]
	for i, doc in enumerate(docs, 1):
		source = doc.metadata.get("source", f"Document {i}")
		content = doc.page_content.strip()
		formatted.append(f"[Source: {source}]\n{content}")
	return "\n\n".join(formatted)


def create_qa_chain(retriever: Any, llm: ChatOpenAI, prompt: PromptTemplate):
	"""
	Set up question answering chain
	"""
	return (
		{
			"context": retriever | format_docs,
			"question": RunnablePassthrough(),
		}
		| prompt
		| llm
		| StrOutputParser()
	)

def create_qr_chain(retriever: Any, llm: ChatOpenAI, prompt: PromptTemplate):
	return (
		{
			"context": retriever | format_docs,
			"question": RunnablePassthrough(),
		}
	)


def get_parser() -> argparse.ArgumentParser:
	"""
	Parse command line arguments
	"""
	parser = argparse.ArgumentParser(description="RAG with vLLM and langchain")

	# Add command line arguments
	parser.add_argument(
		"--vllm-api-key", default="EMPTY", help="API key for vLLM compatible services"
	)
	parser.add_argument(
		"--vllm-embedding-endpoint",
		default="http://localhost:8000/v1",
		help="Base URL for embedding service",
	)
	parser.add_argument(
		"--vllm-chat-endpoint",
		default="http://localhost:8001/v1",
		help="Base URL for chat service",
	)
	parser.add_argument("--uri", default="./milvus.db", help="URI for Milvus database")
	parser.add_argument(
		"--url",
		default=("https://docs.vllm.ai/en/latest/getting_started/quickstart.html"),
		help="URL of the document to process",
	)
	parser.add_argument(
		"--embedding-model",
		default="ssmits/Qwen2-7B-Instruct-embed-base",
		help="Model name for embeddings",
	)
	parser.add_argument(
		"--chat-model", default="qwen/Qwen1.5-0.5B-Chat", help="Model name for chat"
	)
	parser.add_argument(
		"-i", "--interactive", action="store_true", help="Enable interactive Q&A mode"
	)
	parser.add_argument(
		"-k", "--top-k", type=int, default=5, help="Number of top results to retrieve"
	)
	parser.add_argument(
		"-c",
		"--chunk-size",
		type=int,
		default=1000,
		help="Chunk size for document splitting",
	)
	parser.add_argument(
		"-o",
		"--chunk-overlap",
		type=int,
		default=200,
		help="Chunk overlap for document splitting",
	)

	return parser


def init_config(args: Namespace):
	"""
	Initialize configuration settings from command line arguments
	"""
	print(f"config:\nvllm_api_key: {args.vllm_api_key}, \nvllm_embedding_endpoint: {args.vllm_embedding_endpoint}, \nvllm_chat_endpoint: {args.vllm_chat_endpoint}, \nuri: {args.uri}, \nembedding_model: {args.embedding_model}, \nchat_model: {args.chat_model}, \nurl: {args.url}, \nchunk_size: {args.chunk_size}, \nchunk_overlap: {args.chunk_overlap}, \ntop_k: {args.top_k}")
	return {
		"vllm_api_key": args.vllm_api_key,
		"vllm_embedding_endpoint": args.vllm_embedding_endpoint,
		"vllm_chat_endpoint": args.vllm_chat_endpoint,
		"uri": args.uri,
		"embedding_model": args.embedding_model,
		"chat_model": args.chat_model,
		"url": args.url,
		"chunk_size": args.chunk_size,
		"chunk_overlap": args.chunk_overlap,
		"top_k": args.top_k,
	}


def main():
	# Parse command line arguments
	args = get_parser().parse_args()

	# Initialize configuration
	config = init_config(args)

	# Load and split documents
	#documents = load_and_split_documents(config)

	# Initialize vector store and retriever
	#vectorstore = init_vectorstore(config)
	vectorstore = init_milvus(config)

	# kwargs
	retriever = vectorstore.as_retriever(search_kwargs={"k": config["top_k"]})

	# Initialize llm and prompt
	llm = init_llm(config)

	prompt = get_qa_prompt()

	rewrite_prompt = get_qr_prompt()

	# Set up QA chain
	qa_chain = create_qa_chain(retriever, llm, prompt)

	qr_chain = create_qa_chain(retriever, llm, rewrite_prompt)
	# Interactive mode
	if args.interactive:
		print("\nWelcome to Interactive Q&A System!")
		print("Enter 'q' or 'quit' to exit.")

		while True:
			question = input("\nPlease enter your question: ")
			if question.lower() in ["q", "quit"]:
				print("\nThank you for using! Goodbye!")
				break
#			question = qa_chain.invoke(question)
#			print(f"REWRITE : {question}")
			output = qa_chain.invoke(question)
			print(output)
	else:
		# Default single question mode
		question = "How to install vLLM?"
		output = qa_chain.invoke(question)
		print("-" * 50)
		print(output)
		print("-" * 50)


if __name__ == "__main__":
	main()
