from __future__ import annotations

"""
Query Intent Router — rule-based intent classification.

Classifies a natural language query into one of 5 intents:

    SYMBOL_LOOKUP    — "where is validate_token defined?"
    FLOW_EXPLANATION — "explain authentication flow"
    IMPACT_ANALYSIS  — "what breaks if I change validate_token?"
    DEPENDENCY_QUERY — "what does login() depend on?"
    GENERAL_QUERY    — fallback

Also extracts the primary symbol name from the query when detectable.

No ML model used — pure pattern matching with priority ordering.
"""

import re
import logging
from typing import Optional

from app.schemas.models import QueryIntent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pattern definitions
# Priority: patterns are evaluated in order; first match wins.
# Each entry: (intent, list_of_regex_patterns)
# ---------------------------------------------------------------------------

_INTENT_PATTERNS: list[tuple[QueryIntent, list[str]]] = [
    (
        QueryIntent.IMPACT_ANALYSIS,
        [
            r"what (breaks?|changes?|happens?|fails?)\s+(if|when)\s+I\s+(modify|change|remove|delete|update|refactor)",
            r"impact\s+of\s+(changing|modifying|removing|deleting|updating)",
            r"what\s+depends?\s+on\s+\w+",
            r"who\s+(calls?|uses?|imports?)\s+\w+",
            r"side[- ]?effects?\s+of\s+(changing|modifying|removing)",
            r"what\s+would\s+break\s+if",
            r"reverse\s+depend",
            r"affected\s+by\s+(changing|modifying)",
            r"callers?\s+of\s+\w+",
            r"what\s+calls?\s+\w+",
        ],
    ),
    (
        QueryIntent.SYMBOL_LOOKUP,
        [
            r"where\s+is\s+\w+\s+(defined|implemented|declared|located|found)",
            r"where\s+(is|are)\s+\w+",
            r"find\s+(the\s+)?(function|class|method|definition\s+of|implementation\s+of)\s+\w+",
            r"locate\s+(the\s+)?\w+",
            r"show\s+me\s+(the\s+)?(code\s+(for|of)|implementation\s+of|definition\s+of)\s+\w+",
            r"(definition|implementation|declaration)\s+of\s+\w+",
            r"where\s+does\s+\w+\s+(live|exist|reside)",
            r"which\s+file\s+(contains?|has|defines?)\s+\w+",
        ],
    ),
    (
        QueryIntent.FLOW_EXPLANATION,
        [
            r"explain\s+.*\s+flow",
            r"how\s+does\s+.*\s+work",
            r"walk\s+me\s+through",
            r"trace\s+(the\s+)?execution",
            r"what\s+happens\s+when",
            r"execution\s+flow",
            r"explain\s+(the\s+)?(authentication|login|registration|request|response|data|process)",
            r"how\s+(is|are)\s+.*\s+(handled|processed|validated|authenticated)",
            r"describe\s+(the\s+)?\w+\s+(flow|process|pipeline|logic)",
            r"step[- ]?by[- ]?step",
        ],
    ),
    (
        QueryIntent.DEPENDENCY_QUERY,
        [
            r"what\s+does\s+\w+\s+(import|depend\s+on|use|require|need)",
            r"dependencies\s+of\s+\w+",
            r"what\s+(imports?|uses?)\s+\w+",
            r"show\s+(me\s+)?(the\s+)?imports?\s+(of|for|in)\s+\w+",
            r"what\s+modules?\s+does\s+\w+\s+(use|import|depend)",
            r"list\s+dependencies",
            r"dependency\s+(tree|graph|chain)\s+(of|for)\s+\w+",
        ],
    ),
]

# ---------------------------------------------------------------------------
# Symbol extraction patterns — extract the primary symbol name from query
# ---------------------------------------------------------------------------

_SYMBOL_EXTRACTION_PATTERNS: list[str] = [
    r"(?:where\s+is|find|locate|show\s+me\s+(?:the\s+)?(?:code\s+(?:for|of)|implementation\s+of)?|definition\s+of|implementation\s+of)\s+[`'\"]?(\w+)[`'\"]?",
    r"(?:what\s+breaks?\s+if\s+I\s+(?:modify|change|remove|delete)\s+)[`'\"]?(\w+)[`'\"]?",
    r"(?:impact\s+of\s+(?:changing|modifying|removing))\s+[`'\"]?(\w+)[`'\"]?",
    r"(?:what\s+depends?\s+on)\s+[`'\"]?(\w+)[`'\"]?",
    r"(?:what\s+does)\s+[`'\"]?(\w+)[`'\"]?\s+(?:depend|import|use)",
    r"(?:callers?\s+of|who\s+calls?)\s+[`'\"]?(\w+)[`'\"]?",
    r"(?:dependencies\s+of)\s+[`'\"]?(\w+)[`'\"]?",
    r"(?:how\s+does)\s+[`'\"]?(\w+)[`'\"]?\s+(?:work)",
    # Backtick-quoted names: `validate_token`
    r"`(\w+)`",
    # Single or double quoted names: 'validate_token' or "validate_token"
    r"['\"](\w+)['\"]",
]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_intent(query: str) -> tuple[QueryIntent, Optional[str]]:
    """
    Classify a natural language query into a QueryIntent.

    Args:
        query: Raw user query string.

    Returns:
        Tuple of (QueryIntent, Optional[symbol_name]).
        symbol_name is the primary symbol referenced in the query, if detectable.
    """
    normalized = query.strip().lower()

    matched_intent = QueryIntent.GENERAL_QUERY
    for intent, patterns in _INTENT_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, normalized, re.IGNORECASE):
                matched_intent = intent
                break
        if matched_intent != QueryIntent.GENERAL_QUERY:
            break

    # Extract symbol name
    symbol_name = _extract_symbol_name(query)

    logger.debug(
        "Intent classification: '%s' → %s (symbol: %s)",
        query[:80],
        matched_intent.value,
        symbol_name,
    )

    return matched_intent, symbol_name


def _extract_symbol_name(query: str) -> Optional[str]:
    """
    Extract the primary symbol name referenced in a query.

    Tries multiple extraction patterns and returns the first match.
    Returns None if no clear symbol name is found.
    """
    for pattern in _SYMBOL_EXTRACTION_PATTERNS:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            candidate = match.group(1)
            # Filter out common stopwords that might match
            if candidate.lower() not in _STOPWORDS:
                return candidate
    return None


def extract_keywords(query: str) -> list[str]:
    """
    Extract potential symbol name keywords from a query for keyword-based retrieval.

    Tokenizes the query, removes stopwords and short tokens.
    Returns list of candidate symbol names to search in the graph.
    """
    # Remove punctuation except underscores
    cleaned = re.sub(r"[^\w\s]", " ", query)
    tokens = cleaned.split()

    keywords: list[str] = []
    for token in tokens:
        token_lower = token.lower()
        if (
            len(token) >= 3
            and token_lower not in _STOPWORDS
            and not token.isdigit()
            # Include snake_case and camelCase identifiers
        ):
            keywords.append(token)

    # Also extract compound words — look for CamelCase and snake_case
    camel_matches = re.findall(r"\b([A-Z][a-zA-Z]{2,})\b", query)
    snake_matches = re.findall(r"\b([a-z][a-z0-9]{1,}(?:_[a-z][a-z0-9]+)+)\b", query)

    keywords.extend(camel_matches)
    keywords.extend(snake_matches)

    # Deduplicate while preserving order
    seen: set[str] = set()
    result: list[str] = []
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower not in seen:
            seen.add(kw_lower)
            result.append(kw)

    return result


_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", "i", "me", "my", "we", "our",
        "you", "your", "it", "its", "this", "that", "these", "those",
        "what", "where", "when", "how", "why", "who", "which",
        "if", "then", "else", "not", "no", "yes", "all", "any", "each",
        "show", "tell", "find", "get", "give", "make", "let", "use",
        "explain", "describe", "list", "define", "function", "class",
        "method", "module", "file", "code", "flow", "logic",
    }
)
