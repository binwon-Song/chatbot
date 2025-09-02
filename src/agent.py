import openai
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from typing import List, Dict, Annotated
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from dotenv import load_dotenv

@tool
def search_db(query):
	

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
 
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_output_tokens=200,
)
tools = load_tools(["google-serper"],llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)


messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "LLM은 어떤 원리로 작동하나요? 100자 이내로 설명해주세요."},
]


res = agent.run("linux의 가장 최근 release 버전은 무엇이야")
print(res)

