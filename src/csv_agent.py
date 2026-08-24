"""
CSV analysis using LangChain Pandas agent with improved reliability.

Uses a detailed system prefix to ensure the agent performs proper
pandas operations (sum, average, count, etc.) and returns clean answers.
"""

import logging
import re

import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq

logger = logging.getLogger("financial_analyst.csv_agent")

# System prefix to guide the pandas agent
_AGENT_PREFIX = """You are a data analyst working with a pandas DataFrame called `df`.

RULES:
1. ALWAYS use python_repl_ast tool to run pandas code and get the answer.
2. For calculations like sum, average, mean, count, min, max — use df['column'].sum(), df['column'].mean(), etc.
3. For filtering, use df[df['column'] == value] or df.query().
4. For groupby operations, use df.groupby('column')['value_column'].sum() etc.
5. NEVER guess numbers. ALWAYS compute them from the data.
6. Return a clear, concise answer with the actual computed values.
7. If a column name has spaces or special characters, use df['column name'] syntax.
8. When asked about the data structure, use df.shape, df.columns, df.dtypes, df.describe().

Available columns: {columns}
Data types: {dtypes}
Shape: {shape} rows × {ncols} columns
"""


def analyze_csv(question: str, df: pd.DataFrame, llm: ChatGroq) -> str:
    """
    Use a LangChain Pandas agent to answer questions about CSV data.

    Args:
        question: The user's natural-language question.
        df: The DataFrame to analyze.
        llm: The LLM instance.

    Returns:
        The agent's text response.

    Raises:
        RuntimeError: If the agent fails to produce a response.
    """
    logger.info("CSV analysis request: %s", question[:100])

    # Build context about the dataframe
    prefix = _AGENT_PREFIX.format(
        columns=", ".join(df.columns.tolist()),
        dtypes=", ".join(f"{c}: {t}" for c, t in df.dtypes.items()),
        shape=f"{df.shape[0]:,}",
        ncols=df.shape[1],
    )

    try:
        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=False,
            allow_dangerous_code=True,
            prefix=prefix,
            agent_executor_kwargs={
                "handle_parsing_errors": True,
                "max_iterations": 8,
            },
        )
        response = agent.invoke({"input": question})
        answer = response.get("output", "")

        if not answer:
            return "I analyzed the data but couldn't generate a clear answer. Could you rephrase your question?"

        # Strip <think>...</think> reasoning blocks from model output
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()

        logger.info("CSV analysis completed successfully")
        return answer

    except Exception as e:
        logger.error("CSV analysis failed: %s", e)
        # Try a simple direct pandas fallback for common operations
        fallback = _try_direct_pandas(question, df)
        if fallback:
            return fallback
        raise RuntimeError(
            "I had trouble analyzing the CSV data. Please try rephrasing your question.\n\n"
            f"Technical details: {e}"
        ) from e


def _try_direct_pandas(question: str, df: pd.DataFrame) -> str:
    """Fallback: try to answer common questions directly with pandas."""
    q = question.lower()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    try:
        # Sum
        if any(w in q for w in ["total", "sum"]):
            for col in numeric_cols:
                if col.lower() in q:
                    val = df[col].sum()
                    return f"The total **{col}** is **{val:,.2f}**"
            if numeric_cols:
                col = numeric_cols[0]
                val = df[col].sum()
                return f"The total **{col}** is **{val:,.2f}**"

        # Average / Mean
        if any(w in q for w in ["average", "mean", "avg"]):
            for col in numeric_cols:
                if col.lower() in q:
                    val = df[col].mean()
                    return f"The average **{col}** is **{val:,.2f}**"
            if numeric_cols:
                col = numeric_cols[0]
                val = df[col].mean()
                return f"The average **{col}** is **{val:,.2f}**"

        # Count
        if any(w in q for w in ["how many", "count", "number of"]):
            return f"The dataset has **{len(df):,}** rows and **{len(df.columns)}** columns."

        # Max / Min
        if any(w in q for w in ["maximum", "highest", "max", "largest", "top"]):
            for col in numeric_cols:
                if col.lower() in q:
                    val = df[col].max()
                    return f"The maximum **{col}** is **{val:,.2f}**"

        if any(w in q for w in ["minimum", "lowest", "min", "smallest"]):
            for col in numeric_cols:
                if col.lower() in q:
                    val = df[col].min()
                    return f"The minimum **{col}** is **{val:,.2f}**"

    except Exception:
        pass

    return ""
