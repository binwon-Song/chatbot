from langchain_core.documents import Document
import os
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_openai import OpenAIEmbeddings
import dotenv

docs = [
    Document(
        page_content="In 'The Whispering Walls' by Ava Moreno, a young journalist named Sophia uncovers a decades-old conspiracy hidden within the crumbling walls of an ancient mansion, where the whispers of the past threaten to destroy her own sanity.",
        metadata={"category": "Mystery"},
    ),
    Document(
        page_content="In 'The Last Refuge' by Ethan Blackwood, a group of survivors must band together to escape a post-apocalyptic wasteland, where the last remnants of humanity cling to life in a desperate bid for survival.",
        metadata={"category": "Post-Apocalyptic"},
    ),
    Document(
        page_content="In 'The Memory Thief' by Lila Rose, a charismatic thief with the ability to steal and manipulate memories is hired by a mysterious client to pull off a daring heist, but soon finds themselves trapped in a web of deceit and betrayal.",
        metadata={"category": "Heist/Thriller"},
    ),
    Document(
        page_content="In 'The City of Echoes' by Julian Saint Clair, a brilliant detective must navigate a labyrinthine metropolis where time is currency, and the rich can live forever, but at a terrible cost to the poor.",
        metadata={"category": "Science Fiction"},
    ),
    Document(
        page_content="In 'The Starlight Serenade' by Ruby Flynn, a shy astronomer discovers a mysterious melody emanating from a distant star, which leads her on a journey to uncover the secrets of the universe and her own heart.",
        metadata={"category": "Science Fiction/Romance"},
    ),
    Document(
        page_content="In 'The Shadow Weaver' by Piper Redding, a young orphan discovers she has the ability to weave powerful illusions, but soon finds herself at the center of a deadly game of cat and mouse between rival factions vying for control of the mystical arts.",
        metadata={"category": "Fantasy"},
    ),
]


dotenv.load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
URI="./test.db"

vectorstore = Milvus.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(),
    builtin_function=BM25BuiltInFunction(),  # output_field_names="sparse"),
    vector_field=["dense", "sparse"],
    connection_args={
        "uri": URI,
    },
    consistency_level="Bounded",  # Supported values are (`"Strong"`, `"Session"`, `"Bounded"`, `"Eventually"`). See https://milvus.io/docs/consistency.md#Consistency-Level for more details.
    drop_old=False,
)


from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model_name="o1-mini-2024-09-12", temperature=0)

PROMPT_TEMPLATE = """
Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
Use the following pieces of information to provide a concise answer to the question enclosed in <question> tags.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

<question>
{question}
</question>

The response should be specific and use statistics or numbers when possible.

Assistant:"""

prompt = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)
retriever = vectorstore.as_retriever()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query = "What is PAL and PoT?"
res = rag_chain.invoke(query)
print("RESPONSE: ")
print(res)

