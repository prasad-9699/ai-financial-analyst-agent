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
    
    system_prompt = f"""You are an intelligent query router. Determine the BEST route for the user's question.

Available Data Sources:
- CSV Data Loaded: {has_csv}
- PDF Document Loaded: {has_pdf}

Valid Routes:
- 'chart': The user explicitly asks for a visual chart, graph, or plot.
- 'csv': The user asks about data analysis, statistics, rows, columns, or information likely found in a spreadsheet/CSV.
- 'pdf': The user asks about a document, report, or information likely found in the uploaded PDF.
- 'general': The user asks a general knowledge question, greetings, or something clearly unrelated to the uploaded data (e.g., "what is an OS", "who are you").

Respond with EXACTLY ONE word from the valid routes list (chart, csv, pdf, general) in lowercase. Do not include any punctuation or extra text."""

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ])
        decision = response.content.strip().lower()
        
        # Fallback handling based on decision
        if "chart" in decision and has_csv:
            route = Route.CHART
        elif "csv" in decision and has_csv:
            route = Route.CSV
        elif "pdf" in decision and has_pdf:
            route = Route.PDF
        elif "general" in decision:
            route = Route.GENERAL
        else:
            # Safe fallbacks if LLM hallucinates
            if has_csv:
                route = Route.CSV
            elif has_pdf:
                route = Route.PDF
            else:
                route = Route.GENERAL
                
        logger.info("LLM routed question to %s: %s", route.value, question[:80])
        return route
    except Exception as e:
        logger.error("LLM routing failed, falling back to heuristics: %s", e)
        # Fallback to simple heuristic if LLM fails
        q_lower = question.lower()
        if "chart" in q_lower or "graph" in q_lower:
            return Route.CHART if has_csv else Route.GENERAL
        if "pdf" in q_lower or "document" in q_lower:
            return Route.PDF if has_pdf else Route.GENERAL
        return Route.CSV if has_csv else Route.GENERAL
