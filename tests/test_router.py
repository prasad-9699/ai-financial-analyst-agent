"""Tests for question routing logic."""

import pytest
from src.router import Route, route_question

class MockLLM:
    def __init__(self, answer: str):
        self.answer = answer
        
    def invoke(self, messages):
        class Response:
            content = self.answer
        return Response()

class TestRouteQuestion:
    """Test the LLM-based route_question function."""

    def test_chart_request_with_csv(self):
        llm = MockLLM("chart")
        assert route_question(llm, "show me a bar chart", has_csv=True, has_pdf=False) == Route.CHART

    def test_chart_request_without_csv_falls_to_general(self):
        llm = MockLLM("chart") # LLM thinks chart, but no CSV available
        assert route_question(llm, "show me a chart", has_csv=False, has_pdf=False) == Route.GENERAL

    def test_pdf_request_with_pdf(self):
        llm = MockLLM("pdf")
        assert route_question(llm, "what does the document say?", has_csv=False, has_pdf=True) == Route.PDF

    def test_csv_request_with_csv(self):
        llm = MockLLM("csv")
        assert route_question(llm, "what is the total revenue?", has_csv=True, has_pdf=False) == Route.CSV

    def test_general_request_with_data(self):
        llm = MockLLM("general")
        assert route_question(llm, "explain P/E ratio", has_csv=True, has_pdf=True) == Route.GENERAL

    def test_no_data_routes_to_general(self):
        llm = MockLLM("csv") # Even if LLM says CSV, no data means general
        assert route_question(llm, "explain P/E ratio", has_csv=False, has_pdf=False) == Route.GENERAL

    def test_llm_hallucination_fallback(self):
        llm = MockLLM("nonsense_word")
        # Should fallback safely to available data
        assert route_question(llm, "what is the average profit?", has_csv=True, has_pdf=False) == Route.CSV
        assert route_question(llm, "what does it say?", has_csv=False, has_pdf=True) == Route.PDF
        assert route_question(llm, "hello", has_csv=False, has_pdf=False) == Route.GENERAL
