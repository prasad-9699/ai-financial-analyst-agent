"""
CSV analysis using LangChain Pandas agent.

Wraps the pandas dataframe agent with proper error handling and timeout.
"""

import logging

import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_groq import ChatGroq

logger = logging.getLogger("financial_analyst.csv_agent")


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

    try:
        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=False,
            allow_dangerous_code=True,
            agent_executor_kwargs={"handle_parsing_errors": True},
        )
        response = agent.invoke({"input": question})
        answer = response.get("output", "")

        if not answer:
            return "I analyzed the data but couldn't generate a clear answer. Could you rephrase your question?"

        logger.info("CSV analysis completed successfully")
        return answer

    except Exception as e:
        logger.error("CSV analysis failed: %s", e)
        raise RuntimeError(
            f"I had trouble analyzing the CSV data. This might be due to a complex query "
            f"or a temporary API issue. Please try rephrasing your question.\n\nTechnical details: {e}"
        ) from e
