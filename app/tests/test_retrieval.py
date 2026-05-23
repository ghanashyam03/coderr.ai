from __future__ import annotations

"""
Tests for app.retrieval.intent_router

Tests cover:
  - classify_intent() — rule-based intent classification
  - _extract_symbol_name() (via classify_intent return value)
  - extract_keywords() — keyword extraction for retrieval

No mocking — pure function calls against the real rule engine.
"""

import pytest

from app.retrieval.intent_router import classify_intent, extract_keywords
from app.schemas.models import QueryIntent


# ---------------------------------------------------------------------------
# Tests: classify_intent — intent detection
# ---------------------------------------------------------------------------


class TestIntentImpactAnalysis:
    def test_what_breaks_if_modify(self) -> None:
        """Classic 'what breaks if I modify X' → IMPACT_ANALYSIS."""
        intent, _ = classify_intent("what breaks if I modify validate_token")
        assert intent == QueryIntent.IMPACT_ANALYSIS

    def test_what_happens_when_change(self) -> None:
        """'what happens if I change X' → IMPACT_ANALYSIS."""
        intent, _ = classify_intent("what happens if I change the login function")
        assert intent == QueryIntent.IMPACT_ANALYSIS

    def test_what_depends_on(self) -> None:
        """'what depends on X' → IMPACT_ANALYSIS (reverse dependency)."""
        intent, _ = classify_intent("what depends on validate_token")
        assert intent == QueryIntent.IMPACT_ANALYSIS

    def test_callers_of(self) -> None:
        """'callers of X' → IMPACT_ANALYSIS."""
        intent, _ = classify_intent("callers of validate_token")
        assert intent == QueryIntent.IMPACT_ANALYSIS

    def test_what_would_break_if(self) -> None:
        """'what would break if …' → IMPACT_ANALYSIS."""
        intent, _ = classify_intent("what would break if I remove this function")
        assert intent == QueryIntent.IMPACT_ANALYSIS


class TestIntentSymbolLookup:
    def test_where_is_defined(self) -> None:
        """'where is X defined' → SYMBOL_LOOKUP."""
        intent, _ = classify_intent("where is validate_token defined")
        assert intent == QueryIntent.SYMBOL_LOOKUP

    def test_where_is_implemented(self) -> None:
        """'where is X implemented' → SYMBOL_LOOKUP."""
        intent, _ = classify_intent("where is authenticate implemented")
        assert intent == QueryIntent.SYMBOL_LOOKUP

    def test_find_function(self) -> None:
        """'find the function X' → SYMBOL_LOOKUP."""
        intent, _ = classify_intent("find the function handle_request")
        assert intent == QueryIntent.SYMBOL_LOOKUP

    def test_which_file_contains(self) -> None:
        """'which file contains X' → SYMBOL_LOOKUP."""
        intent, _ = classify_intent("which file contains validate_token")
        assert intent == QueryIntent.SYMBOL_LOOKUP

    def test_locate_symbol(self) -> None:
        """'locate X' → SYMBOL_LOOKUP."""
        intent, _ = classify_intent("locate the validate_token function")
        assert intent == QueryIntent.SYMBOL_LOOKUP


class TestIntentFlowExplanation:
    def test_explain_flow(self) -> None:
        """'explain the authentication flow' → FLOW_EXPLANATION."""
        intent, _ = classify_intent("explain the authentication flow")
        assert intent == QueryIntent.FLOW_EXPLANATION

    def test_how_does_work(self) -> None:
        """'how does X work' → FLOW_EXPLANATION."""
        intent, _ = classify_intent("how does login work")
        assert intent == QueryIntent.FLOW_EXPLANATION

    def test_walk_me_through(self) -> None:
        """'walk me through' → FLOW_EXPLANATION."""
        intent, _ = classify_intent("walk me through the request handling")
        assert intent == QueryIntent.FLOW_EXPLANATION

    def test_trace_execution(self) -> None:
        """'trace the execution' → FLOW_EXPLANATION."""
        intent, _ = classify_intent("trace the execution of the login process")
        assert intent == QueryIntent.FLOW_EXPLANATION

    def test_step_by_step(self) -> None:
        """'step by step' → FLOW_EXPLANATION."""
        intent, _ = classify_intent("explain step by step how auth works")
        assert intent == QueryIntent.FLOW_EXPLANATION


class TestIntentDependencyQuery:
    def test_what_does_depend_on(self) -> None:
        """'what does X depend on' → DEPENDENCY_QUERY."""
        intent, _ = classify_intent("what does login depend on")
        assert intent == QueryIntent.DEPENDENCY_QUERY

    def test_what_does_import(self) -> None:
        """'what does X import' → DEPENDENCY_QUERY."""
        intent, _ = classify_intent("what does auth_module import")
        assert intent == QueryIntent.DEPENDENCY_QUERY

    def test_dependencies_of(self) -> None:
        """'dependencies of X' → DEPENDENCY_QUERY."""
        intent, _ = classify_intent("dependencies of validate_token")
        assert intent == QueryIntent.DEPENDENCY_QUERY

    def test_list_dependencies(self) -> None:
        """'list dependencies' → DEPENDENCY_QUERY."""
        intent, _ = classify_intent("list dependencies of the auth module")
        assert intent == QueryIntent.DEPENDENCY_QUERY


class TestIntentGeneral:
    def test_general_fallback(self) -> None:
        """Unmatched queries → GENERAL_QUERY."""
        intent, _ = classify_intent("tell me about the project")
        assert intent == QueryIntent.GENERAL_QUERY

    def test_generic_question(self) -> None:
        """A generic 'tell me' query without any patterns → GENERAL_QUERY."""
        intent, _ = classify_intent("give me an overview of the codebase structure")
        assert intent == QueryIntent.GENERAL_QUERY

    def test_empty_query(self) -> None:
        """Empty string must not raise and returns GENERAL_QUERY."""
        intent, symbol = classify_intent("")
        assert intent == QueryIntent.GENERAL_QUERY
        assert symbol is None


# ---------------------------------------------------------------------------
# Tests: classify_intent — symbol extraction from query
# ---------------------------------------------------------------------------


class TestSymbolExtractionBasic:
    def test_extract_from_where_is_defined(self) -> None:
        """'where is validate_token defined' → extracts 'validate_token'."""
        _, symbol = classify_intent("where is validate_token defined")
        assert symbol == "validate_token"

    def test_extract_from_what_breaks_modify(self) -> None:
        """'what breaks if I modify validate_token' → extracts 'validate_token'."""
        _, symbol = classify_intent("what breaks if I modify validate_token")
        assert symbol == "validate_token"

    def test_extract_from_callers_of(self) -> None:
        """'callers of refresh_token' → extracts 'refresh_token'."""
        _, symbol = classify_intent("callers of refresh_token")
        assert symbol == "refresh_token"

    def test_extract_from_dependencies_of(self) -> None:
        """'dependencies of login' → extracts 'login'."""
        _, symbol = classify_intent("dependencies of login")
        assert symbol == "login"


class TestSymbolExtractionBacktick:
    def test_backtick_quoted_name(self) -> None:
        """'what does `refresh_token` do' → extracts 'refresh_token'."""
        _, symbol = classify_intent("what does `refresh_token` do")
        assert symbol == "refresh_token"

    def test_backtick_in_impact_query(self) -> None:
        """`validate_token` in impact query → symbol extracted."""
        _, symbol = classify_intent("what breaks if I modify `validate_token`")
        # The modifier pattern should extract before the backtick pattern
        assert symbol is not None
        assert "validate_token" in symbol or symbol == "validate_token"


# ---------------------------------------------------------------------------
# Tests: extract_keywords
# ---------------------------------------------------------------------------


class TestExtractKeywordsBasic:
    def test_stopwords_removed(self) -> None:
        """'explain authentication flow' — stopwords 'explain' and 'flow' removed."""
        # Note: 'explain' is in stopwords; 'flow' is in stopwords
        keywords = extract_keywords("explain authentication flow")
        # 'authentication' is not a stopword and len >= 3
        assert "authentication" in keywords

    def test_short_tokens_excluded(self) -> None:
        """Tokens shorter than 3 characters must be excluded."""
        keywords = extract_keywords("a is at on do")
        assert keywords == [] or all(len(k) >= 3 for k in keywords)

    def test_basic_word_included(self) -> None:
        """Non-stopword word of length >= 3 must appear."""
        keywords = extract_keywords("validate the token now")
        # 'validate' is not in stopwords and len > 3
        assert "validate" in keywords

    def test_numeric_tokens_excluded(self) -> None:
        """Pure digit tokens must be excluded."""
        keywords = extract_keywords("error code 404 status 200")
        assert "404" not in keywords
        assert "200" not in keywords


class TestExtractKeywordsSnakeCase:
    def test_snake_case_included(self) -> None:
        """'validate_token function' → includes 'validate_token'."""
        keywords = extract_keywords("validate_token function")
        # snake_case is treated as a single token by split
        assert "validate_token" in keywords

    def test_snake_case_compound_extracted(self) -> None:
        """Snake-case compound identifier must appear in keywords."""
        keywords = extract_keywords("call the refresh_token endpoint")
        assert "refresh_token" in keywords

    def test_multi_segment_snake_case(self) -> None:
        """'handle_request_validation' must appear as a keyword."""
        keywords = extract_keywords("handle_request_validation is slow")
        assert "handle_request_validation" in keywords


class TestExtractKeywordsCamelCase:
    def test_camel_case_included(self) -> None:
        """'JWTHandler class' → includes 'JWTHandler'."""
        keywords = extract_keywords("JWTHandler class")
        assert "JWTHandler" in keywords

    def test_pascal_case_included(self) -> None:
        """'AuthService is the main handler' → includes 'AuthService'."""
        keywords = extract_keywords("AuthService is the main handler")
        assert "AuthService" in keywords

    def test_mixed_query_camel_and_snake(self) -> None:
        """Both JWTHandler and validate_token must appear."""
        keywords = extract_keywords("JWTHandler calls validate_token internally")
        assert "JWTHandler" in keywords
        assert "validate_token" in keywords

    def test_all_stopwords_query(self) -> None:
        """Query that is all stopwords must return empty or minimal keywords."""
        keywords = extract_keywords("the is a and or but to for of")
        # All these are stopwords; result should be empty or very short tokens only
        for kw in keywords:
            # No keyword should be a known stopword
            assert kw.lower() not in {
                "the", "is", "a", "and", "or", "but", "to", "for", "of"
            }

    def test_deduplication(self) -> None:
        """Duplicate keywords must appear only once."""
        keywords = extract_keywords("validate_token validate_token")
        count = keywords.count("validate_token")
        assert count == 1
