"""
Question routing logic.

Determines which tool to use based on the user's question and available data sources.
"""

import logging
from enum import Enum

logger = logging.getLogger("financial_analyst.router")


class Route(str, Enum):
    """Available routing destinations."""
    CHART = "chart"
    CSV = "csv"
    PDF = "pdf"
    GENERAL = "general"


from langchain_groq import ChatGroq

def route_question(llm: ChatGroq, question: str, has_csv: bool, has_pdf: bool) -> Route:
    """
    Route a user question using the LLM to understand intent.

    The router strongly prefers uploaded data sources. It only routes to
    'general' when the question is clearly unrelated (greetings, general
    knowledge like "what is an OS", etc.).

    Args:
        llm: The ChatGroq instance for fast classification.
        question: The user's natural-language question.
        has_csv: Whether CSV data is loaded.
        has_pdf: Whether a PDF vector store is available.

    Returns:
        The appropriate Route enum value.
    """
    # If no data is loaded, it must be general
    if not has_csv and not has_pdf:
        return Route.GENERAL

    # Fast intent classification using the LLM
    from langchain_core.messages import SystemMessage, HumanMessage

    # Build context about what's available
    sources = []
    if has_csv:
        sources.append("CSV spreadsheet data")
    if has_pdf:
        sources.append("PDF document")
    sources_str = " and ".join(sources)

    system_prompt = f"""You are an intelligent query router. The user has uploaded: {sources_str}.

CRITICAL RULE: Since the user has uploaded data, MOST questions are about that data. Only route to 'general' if the question is CLEARLY unrelated to any uploaded data (e.g., greetings like "hi/hello", or pure general knowledge like "what is an OS", "who is the president").

If the question mentions ANY name, topic, skill, detail, or asks about ANYTHING that COULD be in the uploaded data — route to the data source, NOT 'general'.

Valid Routes:
- 'chart': The user asks for a visual chart, graph, or plot (requires CSV).
- 'csv': The user asks about data, numbers, statistics, rows, columns, or analysis of spreadsheet data.
- 'pdf': The user asks about content, people, topics, skills, details, or any information that could be in the uploaded PDF document.
- 'general': ONLY for greetings, or questions clearly about general world knowledge that have NOTHING to do with uploaded data.

When in doubt, ALWAYS prefer '{("pdf" if has_pdf and not has_csv else "csv" if has_csv and not has_pdf else "pdf")}' over 'general'.

Respond with EXACTLY ONE word: chart, csv, pdf, or general."""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ])
        raw = response.content
        # Strip <think>...</think> reasoning blocks from model output
        import re
        cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        decision = cleaned.strip().lower()

        # Parse LLM decision
        if "chart" in decision and has_csv:
            route = Route.CHART
        elif "csv" in decision and has_csv:
            route = Route.CSV
        elif "pdf" in decision and has_pdf:
            route = Route.PDF
        elif "general" in decision:
            route = Route.GENERAL
        else:
            # LLM hallucinated — fallback to loaded data
            if has_pdf:
                route = Route.PDF
            elif has_csv:
                route = Route.CSV
            else:
                route = Route.GENERAL

        # Smart override: if LLM said "general" but data IS loaded,
        # check if the question is truly general (greetings / world knowledge)
        if route == Route.GENERAL and (has_csv or has_pdf):
            if not _is_clearly_general(question):
                # Override to the uploaded data source
                if has_pdf and not has_csv:
                    route = Route.PDF
                elif has_csv and not has_pdf:
                    route = Route.CSV
                elif has_pdf:
                    # Both loaded — prefer PDF for non-data questions
                    route = Route.PDF
                logger.info("Overrode 'general' → %s (question likely about uploaded data)", route.value)

        logger.info("LLM routed question to %s: %s", route.value, question[:80])
        return route
    except Exception as e:
        logger.error("LLM routing failed, falling back to heuristics: %s", e)
        # Fallback: prefer loaded data sources
        q_lower = question.lower()
        if has_csv and any(w in q_lower for w in ["chart", "graph", "plot", "visualize"]):
            return Route.CHART
        if has_pdf:
            return Route.PDF
        return Route.CSV if has_csv else Route.GENERAL


def _is_clearly_general(question: str) -> bool:
    """
    Check if a question is clearly general knowledge / greeting,
    i.e., definitely NOT about uploaded data.
    """
    q = question.lower().strip()

    # Greetings
    greetings = ["hi", "hello", "hey", "good morning", "good evening",
                 "good afternoon", "howdy", "what's up", "sup"]
    if q in greetings or any(q.startswith(g + " ") for g in greetings[:3]):
        # Allow "hi" but not "hi, tell me about priya"
        if len(q.split()) <= 3:
            return True

    # Meta questions about the bot
    meta = ["who are you", "what are you", "what can you do", "help",
            "how do you work", "what is this"]
    if any(q.startswith(m) for m in meta):
        return True

    # Pure general knowledge patterns
    general_patterns = [
        "what is a ", "what is an ", "what are ",
        "define ", "explain what ", "who is the president",
        "capital of ", "what is the capital",
    ]
    if any(q.startswith(p) for p in general_patterns):
        # But NOT if it could reference uploaded data, e.g., "what is the revenue"
        data_hints = ["revenue", "profit", "salary", "score", "total", "average",
                      "column", "row", "data", "table", "report", "summary",
                      "skill", "experience", "education", "project"]
        if not any(h in q for h in data_hints):
            return True

    return False
