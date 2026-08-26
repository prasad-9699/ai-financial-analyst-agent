"""
CSV analysis using LangChain Pandas agent with improved reliability.

Uses a detailed system prefix with data preview to ensure the agent performs
proper pandas operations (sum, average, count, etc.) and returns clean answers.
"""

import logging
import re

import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq

logger = logging.getLogger("financial_analyst.csv_agent")

# System prefix to guide the pandas agent — includes a data preview
# so the LLM can see actual values and column names
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
9. Format large numbers with commas (e.g., 1,000,000).
10. Format percentages with % sign and 2 decimal places.
11. When comparing values, show both the value and the label/category.

Available columns: {columns}
Data types: {dtypes}
Shape: {shape} rows × {ncols} columns

Sample data (first 3 rows):
{preview}
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

    # Build a compact text preview of the first 3 rows
    try:
        preview = df.head(3).to_string(index=False, max_colwidth=30)
    except Exception:
        preview = str(df.head(3))

    # Build context about the dataframe
    prefix = _AGENT_PREFIX.format(
        columns=", ".join(df.columns.tolist()),
        dtypes=", ".join(f"{c}: {t}" for c, t in df.dtypes.items()),
        shape=f"{df.shape[0]:,}",
        ncols=df.shape[1],
        preview=preview,
    )

    try:
        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=False,
            allow_dangerous_code=True,
            prefix=prefix,
            max_iterations=8,
            agent_executor_kwargs={
                "handle_parsing_errors": True,
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

        # Build helpful error with column names
        col_list = ", ".join(f"`{c}`" for c in df.columns[:10])
        extra = "..." if len(df.columns) > 10 else ""
        raise RuntimeError(
            "I had trouble analyzing the CSV data. Please try rephrasing your question.\n\n"
            f"**Available columns:** {col_list}{extra}\n\n"
            f"Technical details: {e}"
        ) from e


def _fmt(val, is_pct=False):
    """Format a numeric value nicely."""
    if is_pct:
        return f"{val:.2f}%"
    if isinstance(val, float):
        if abs(val) >= 1_000:
            return f"{val:,.2f}"
        return f"{val:.4f}"
    if isinstance(val, int):
        return f"{val:,}"
    return str(val)


def _try_direct_pandas(question: str, df: pd.DataFrame) -> str:
    """Fallback: try to answer common questions directly with pandas."""
    q = question.lower()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    text_cols = df.select_dtypes(include="object").columns.tolist()

    try:
        # ── Describe / Summary ──────────────────────────────────
        if any(w in q for w in ["describe", "summary", "statistics", "overview", "info"]):
            if numeric_cols:
                desc = df[numeric_cols].describe().round(2)
                return f"**Dataset Summary** ({df.shape[0]:,} rows × {df.shape[1]} columns)\n\n{desc.to_markdown()}"
            return f"The dataset has **{df.shape[0]:,}** rows and **{df.shape[1]}** columns.\n\nColumns: {', '.join(df.columns)}"

        # ── Columns / Structure ─────────────────────────────────
        if any(w in q for w in ["columns", "fields", "what data", "structure"]):
            parts = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                nulls = df[col].isna().sum()
                null_str = f" ({nulls} nulls)" if nulls > 0 else ""
                parts.append(f"- **{col}** — `{dtype}`{null_str}")
            return "**Columns:**\n" + "\n".join(parts)

        # ── Sum / Total ────────────────────────────────────────
        if any(w in q for w in ["total", "sum"]):
            for col in numeric_cols:
                if col.lower() in q:
                    val = df[col].sum()
                    return f"The total **{col}** is **{_fmt(val)}**"
            if numeric_cols:
                col = numeric_cols[0]
                val = df[col].sum()
                return f"The total **{col}** is **{_fmt(val)}**"

        # ── Average / Mean ──────────────────────────────────────
        if any(w in q for w in ["average", "mean", "avg"]):
            for col in numeric_cols:
                if col.lower() in q:
                    val = df[col].mean()
                    return f"The average **{col}** is **{_fmt(val)}**"
            if numeric_cols:
                col = numeric_cols[0]
                val = df[col].mean()
                return f"The average **{col}** is **{_fmt(val)}**"

        # ── Count ───────────────────────────────────────────────
        if any(w in q for w in ["how many", "count", "number of"]):
            return f"The dataset has **{len(df):,}** rows and **{len(df.columns)}** columns."

        # ── Max / Highest / Top ─────────────────────────────────
        if any(w in q for w in ["maximum", "highest", "max", "largest", "top"]):
            # Check for "top N" pattern
            top_match = re.search(r"top\s+(\d+)", q)
            if top_match and numeric_cols:
                n = int(top_match.group(1))
                for col in numeric_cols:
                    if col.lower() in q:
                        top_df = df.nlargest(n, col)
                        return f"**Top {n} by {col}:**\n\n{top_df.to_markdown(index=False)}"
                # Default to first numeric column
                col = numeric_cols[0]
                top_df = df.nlargest(n, col)
                return f"**Top {n} by {col}:**\n\n{top_df.to_markdown(index=False)}"

            for col in numeric_cols:
                if col.lower() in q:
                    val = df[col].max()
                    return f"The maximum **{col}** is **{_fmt(val)}**"

        # ── Min / Lowest ────────────────────────────────────────
        if any(w in q for w in ["minimum", "lowest", "min", "smallest", "bottom"]):
            for col in numeric_cols:
                if col.lower() in q:
                    val = df[col].min()
                    return f"The minimum **{col}** is **{_fmt(val)}**"

        # ── Unique values ───────────────────────────────────────
        if any(w in q for w in ["unique", "distinct", "categories"]):
            for col in text_cols:
                if col.lower() in q:
                    vals = df[col].unique().tolist()
                    return f"**Unique values in {col}** ({len(vals)} total):\n" + ", ".join(str(v) for v in vals[:30])
            if text_cols:
                col = text_cols[0]
                vals = df[col].unique().tolist()
                return f"**Unique values in {col}** ({len(vals)} total):\n" + ", ".join(str(v) for v in vals[:30])

        # ── Correlation ─────────────────────────────────────────
        if any(w in q for w in ["correlation", "correlated", "relationship"]):
            if len(numeric_cols) >= 2:
                corr = df[numeric_cols].corr().round(3)
                return f"**Correlation Matrix:**\n\n{corr.to_markdown()}"

        # ── Group by ────────────────────────────────────────────
        if "by" in q and any(w in q for w in ["group", "each", "per", "breakdown"]):
            group_col = None
            agg_col = None
            for col in text_cols:
                if col.lower() in q:
                    group_col = col
                    break
            for col in numeric_cols:
                if col.lower() in q:
                    agg_col = col
                    break
            if group_col and agg_col:
                result = df.groupby(group_col)[agg_col].sum().sort_values(ascending=False).reset_index()
                return f"**{agg_col} by {group_col}:**\n\n{result.to_markdown(index=False)}"
            elif group_col and numeric_cols:
                agg_col = numeric_cols[0]
                result = df.groupby(group_col)[agg_col].sum().sort_values(ascending=False).reset_index()
                return f"**{agg_col} by {group_col}:**\n\n{result.to_markdown(index=False)}"

    except Exception:
        pass

    return ""
