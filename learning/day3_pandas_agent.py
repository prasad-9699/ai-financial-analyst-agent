import os
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent

load_dotenv()

# ── Setup LLM ─────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# ── Load CSV ──────────────────────────────────────────
csv_path = "data/sample_sales.csv"
df = pd.read_csv(csv_path)

print("=" * 55)
print("  DAY 3 - Pandas Agent Test")
print("=" * 55)
print(f"CSV Loaded: {csv_path}")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")
print("=" * 55)

# ── Create Pandas Agent ───────────────────────────────
agent = create_pandas_dataframe_agent(
    llm=llm,
    df=df,
    verbose=True,             # Shows agent thinking process
    allow_dangerous_code=True, # Allow pandas code execution
    agent_executor_kwargs={
        "handle_parsing_errors": True  # Don't crash on errors
    }
)

# ── Test Questions ────────────────────────────────────
questions = [
    "What is the total revenue across all months?",
    "Which month had the highest revenue?",
    "Which product has the highest total profit?",
    "What is the average revenue per region?",
    "In which month did revenue drop the most?"
]

for i, question in enumerate(questions, 1):
    print(f"\nQ{i}: {question}")
    print("-" * 45)
    try:
        response = agent.invoke({"input": question})
        print(f"Answer: {response['output']}")
    except Exception as e:
        print(f"Error: {e}")
    print("=" * 55)

print("\n Day 3 Complete!")
print(" Agent is reading real CSV and writing Pandas code automatically!")