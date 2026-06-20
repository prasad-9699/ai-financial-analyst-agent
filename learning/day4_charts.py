import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.tools import tool

load_dotenv()

# ── Setup LLM ─────────────────────────────────────────
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# ── Load CSV ──────────────────────────────────────────
df = pd.read_csv("data/sample_sales.csv")

print("=" * 55)
print("  DAY 4 - Chart Generation + Pandas Agent")
print("=" * 55)
print(f"CSV Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
print("=" * 55)

# ══════════════════════════════════════════════════════
# TOOLS — Yeh agent ke weapons hain
# ══════════════════════════════════════════════════════

@tool
def generate_bar_chart(x_column: str, y_column: str) -> str:
    """
    Use this tool to generate a bar chart.
    Input should be column names from the CSV.
    Example: x_column='Month', y_column='Revenue'
    """
    try:
        grouped = df.groupby(x_column)[y_column].sum().reset_index()
        fig = px.bar(
            grouped,
            x=x_column,
            y=y_column,
            title=f"{y_column} by {x_column}",
            color=y_column,
            color_continuous_scale="Blues"
        )
        fig.show()
        return f"Bar chart of {y_column} by {x_column} generated successfully!"
    except Exception as e:
        return f"Chart error: {e}"


@tool
def generate_line_chart(x_column: str, y_column: str) -> str:
    """
    Use this tool to generate a line chart to show trends over time.
    Example: x_column='Month', y_column='Revenue'
    """
    try:
        grouped = df.groupby(x_column)[y_column].sum().reset_index()
        fig = px.line(
            grouped,
            x=x_column,
            y=y_column,
            title=f"{y_column} Trend by {x_column}",
            markers=True
        )
        fig.show()
        return f"Line chart of {y_column} trend generated successfully!"
    except Exception as e:
        return f"Chart error: {e}"


@tool
def generate_pie_chart(names_column: str, values_column: str) -> str:
    """
    Use this tool to generate a pie chart to show distribution or share.
    Example: names_column='Product', values_column='Revenue'
    """
    try:
        grouped = df.groupby(names_column)[values_column].sum().reset_index()
        fig = px.pie(
            grouped,
            names=names_column,
            values=values_column,
            title=f"{values_column} Distribution by {names_column}"
        )
        fig.show()
        return f"Pie chart of {values_column} by {names_column} generated successfully!"
    except Exception as e:
        return f"Chart error: {e}"


@tool
def analyze_data(question: str) -> str:
    """
    Use this tool for any data analysis, calculations, statistics,
    or questions about the CSV data that don't require a chart.
    Example: 'What is total revenue?', 'Which product has highest profit?'
    """
    try:
        pandas_agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=False,
            allow_dangerous_code=True,
            agent_executor_kwargs={"handle_parsing_errors": True}
        )
        response = pandas_agent.invoke({"input": question})
        return response['output']
    except Exception as e:
        return f"Analysis error: {e}"


# ══════════════════════════════════════════════════════
# TEST ALL TOOLS DIRECTLY
# ══════════════════════════════════════════════════════

print("\n[TEST 1] Bar Chart — Monthly Revenue")
print("-" * 45)
result = generate_bar_chart.invoke({"x_column": "Month", "y_column": "Revenue"})
print(f"Result: {result}")

print("\n[TEST 2] Line Chart — Monthly Profit Trend")
print("-" * 45)
result = generate_line_chart.invoke({"x_column": "Month", "y_column": "Profit"})
print(f"Result: {result}")

print("\n[TEST 3] Pie Chart — Revenue by Product")
print("-" * 45)
result = generate_pie_chart.invoke({"names_column": "Product", "values_column": "Revenue"})
print(f"Result: {result}")

print("\n[TEST 4] Data Analysis — No Chart Needed")
print("-" * 45)
result = analyze_data.invoke({"question": "Which region has the highest total profit?"})
print(f"Result: {result}")

print("\n" + "=" * 55)
print(" Day 4 Complete!")
print(" 4 tools created — Bar, Line, Pie charts + Data Analysis!")
print(" Each tool has @tool decorator with clear description")
print(" Agent will use these tools autonomously in Day 6 app!")
print("=" * 55)