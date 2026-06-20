import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
api_key = os.getenv("GROQ_API_KEY")
print("API Key found:", api_key[:8])
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=api_key)
messages = [SystemMessage(content="You are a senior financial analyst AI"), HumanMessage(content="Introduce yourself and list 3 key financial metrics for sales analysis")]
response = llm.invoke(messages)
print("AI RESPONSE:")
print(response.content)
