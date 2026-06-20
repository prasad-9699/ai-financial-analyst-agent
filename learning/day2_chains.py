import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"))

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Senior Financial Analyst AI. Analyze data and give clear bullet point insights."),
    ("human", "Company: {company_name}\nData: {data_summary}\n\nGive 3 key insights, 1 concern, 1 recommendation.")
])

chain = prompt | llm | StrOutputParser()

print("=" * 50)
print("  DAY 2 - Chains Test")
print("=" * 50)

result = chain.invoke({
    "company_name": "TechCorp India Sales 2024",
    "data_summary": "Total Revenue: 2.4 Crore, Best Month: August, Worst Month: September with 35 percent drop, Top Product: Electronics contributing 58 percent revenue"
})

print(result)
print("=" * 50)
print("Day 2 Complete!")
