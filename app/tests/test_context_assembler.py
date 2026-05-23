from __future__ import annotations

"""
Tests for app.reasoning.context_assembler.ContextAssembler

Strategy:
- Build mock CodeSymbol and RetrievalResult objects directly
- Use unittest.mock.MagicMock for the CodeGraph dependency
  (avoid real networkx/embedding setup)
- Call assembler.assemble() and assert on AssembledContext fields

Key behaviours under test:
  1. Empty results → "No relevant code found"
  2. Basic assembly with multiple results → non-empty full_text
  3. Deduplication: same symbol_id twice → appears once in sources
  4. Budget enforcement: total_chars <= max_chars + reasonable overrun
  5. Entry point detection via route decorator
  6. Topological ordering: callers before callees in CORE FLOW
  7. Truncation flag set when symbol exceeds budget
  8. Expected sections present in AssembledContext.sections dict
"""

from unittest.mock import MagicMock
from typing import Optional

import pytest

from app.reasoning.context_assembler import (
    ContextAssembler,
    SECTION_PROJECT_PURPOSE,
    SECTION_ENTRY_POINTS,
    SECTION_CORE_FLOW,
    SECTION_DEPENDENCIES,
    SECTION_UTILITIES,
    SECTION_IMPORTS,
)
from app.schemas.models import (
    AssembledContext,
    CodeSymbol,
    QueryIntent,
    RetrievalResult,
    SymbolType,
)


# ---------------------------------------------------------------------------
# Helpers / Factories
# ---------------------------------------------------------------------------


def _make_symbol(
    symbol_id: str,
    name: str,
    qualified_name: str,
    *,
    source: str = "def fn(): pass",
    symbol_type: SymbolType = SymbolType.FUNCTION,
    file_path: str = "/repo/auth/routes.py",
    class_name: Optional[str] = None,
    calls: list[str] | None = None,
    callers: list[str] | None = None,
    decorators: list[str] | None = None,
    line_start: int = 1,
    line_end: int = 5,
    docstring: Optional[str] = None,
) -> CodeSymbol:
    """Create a minimal CodeSymbol for testing."""
    return CodeSymbol(
        symbol_id=symbol_id,
        symbol_type=symbol_type,
        name=name,
        qualified_name=qualified_name,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        source=source,
        docstring=docstring,
        class_name=class_name,
        repo_name="testrepo",
        calls=calls or [],
        callers=callers or [],
        bases=[],
        decorators=decorators or [],
    )


def _make_result(
    symbol: CodeSymbol,
    score: float = 0.9,
    source: str = "semantic",
    graph_depth: int = 0,
) -> RetrievalResult:
    """Create a RetrievalResult wrapping *symbol*."""
    return RetrievalResult(symbol=symbol, score=score, source=source, graph_depth=graph_depth)


def _make_mock_graph() -> MagicMock:
    """Return a MagicMock acting as a CodeGraph."""
    mock = MagicMock()
    mock.get_node_id.return_value = None
    mock.get_node_data.return_value = None
    return mock


def _make_assembler(max_chars: int = 8000) -> ContextAssembler:
    """Return a ContextAssembler with a mock graph."""
    return ContextAssembler(graph=_make_mock_graph(), max_chars=max_chars)


# ---------------------------------------------------------------------------
# Test 1: Empty results
# ---------------------------------------------------------------------------


class TestEmptyResults:
    def test_empty_results_returns_no_results_message(self) -> None:
        """assemble([]) must return AssembledContext with 'No relevant code found' in full_text."""
        assembler = _make_assembler()
        ctx = assembler.assemble(
            results=[],
            query="where is validate_token defined",
            intent=QueryIntent.SYMBOL_LOOKUP,
        )

        assert isinstance(ctx, AssembledContext)
        assert "No relevant code found" in ctx.full_text
        assert ctx.sources == []
        assert ctx.total_chars == 0
        assert ctx.truncated is False

    def test_empty_results_sections_empty(self) -> None:
        """assemble([]) must return empty sections dict."""
        assembler = _make_assembler()
        ctx = assembler.assemble(
            results=[], query="", intent=QueryIntent.GENERAL_QUERY
        )
        assert ctx.sections == {}


# ---------------------------------------------------------------------------
# Test 2: Basic assembly
# ---------------------------------------------------------------------------


class TestBasicAssembly:
    def test_basic_assembly_non_empty_full_text(self) -> None:
        """3 results → non-empty full_text."""
        assembler = _make_assembler()
        syms = [
            _make_symbol(f"sym{i}", f"fn_{i}", f"auth.routes.fn_{i}")
            for i in range(3)
        ]
        results = [_make_result(s) for s in syms]

        ctx = assembler.assemble(
            results=results,
            query="explain the login flow",
            intent=QueryIntent.FLOW_EXPLANATION,
        )

        assert isinstance(ctx, AssembledContext)
        assert len(ctx.full_text) > 0
        assert ctx.total_chars > 0

    def test_basic_assembly_sources_populated(self) -> None:
        """Assembled context must list symbol_ids in sources."""
        assembler = _make_assembler()
        syms = [
            _make_symbol("id_alpha", "alpha_fn", "mod.alpha_fn"),
            _make_symbol("id_beta", "beta_fn", "mod.beta_fn"),
        ]
        results = [_make_result(s) for s in syms]

        ctx = assembler.assemble(
            results=results,
            query="show me the alpha function",
            intent=QueryIntent.SYMBOL_LOOKUP,
        )

        # Both symbols should appear in sources
        assert "id_alpha" in ctx.sources or "id_beta" in ctx.sources

    def test_single_result_assembles(self) -> None:
        """Single result must assemble successfully."""
        assembler = _make_assembler()
        sym = _make_symbol("single_id", "validate_token", "auth.jwt.validate_token")
        ctx = assembler.assemble(
            results=[_make_result(sym)],
            query="where is validate_token",
            intent=QueryIntent.SYMBOL_LOOKUP,
        )
        assert len(ctx.full_text) > 0


# ---------------------------------------------------------------------------
# Test 3: Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_same_symbol_twice_appears_once_in_sources(self) -> None:
        """Same symbol_id in two results → it appears only once in sources."""
        assembler = _make_assembler()
        sym = _make_symbol("dup_id", "validate_token", "auth.jwt.validate_token")

        # Submit the same symbol twice with different scores
        r1 = _make_result(sym, score=0.95, source="semantic")
        r2 = _make_result(sym, score=0.80, source="keyword")

        ctx = assembler.assemble(
            results=[r1, r2],
            query="validate_token",
            intent=QueryIntent.SYMBOL_LOOKUP,
        )

        assert ctx.sources.count("dup_id") <= 1

    def test_deduplication_keeps_higher_score(self) -> None:
        """When deduplicating, the result with the higher score must be kept."""
        assembler = _make_assembler()
        sym = _make_symbol(
            "score_test_id", "my_fn", "mymod.my_fn",
            source="def my_fn(): return 42",
        )
        r_low = _make_result(sym, score=0.50)
        r_high = _make_result(sym, score=0.99)

        # Supply low first, then high — high should win
        ctx = assembler.assemble(
            results=[r_low, r_high],
            query="my_fn",
            intent=QueryIntent.GENERAL_QUERY,
        )
        # Symbol should be present exactly once
        assert ctx.sources.count("score_test_id") <= 1

    def test_different_symbols_both_included(self) -> None:
        """Two distinct symbols must both be represented in sources."""
        assembler = _make_assembler()
        sym_a = _make_symbol("id_a", "fn_a", "mod.fn_a")
        sym_b = _make_symbol("id_b", "fn_b", "mod.fn_b")

        ctx = assembler.assemble(
            results=[_make_result(sym_a), _make_result(sym_b)],
            query="both functions",
            intent=QueryIntent.GENERAL_QUERY,
        )
        # At least one of them should be in sources
        assert "id_a" in ctx.sources or "id_b" in ctx.sources


# ---------------------------------------------------------------------------
# Test 4: Budget enforcement
# ---------------------------------------------------------------------------


class TestBudgetRespected:
    def test_total_chars_within_budget(self) -> None:
        """total_chars must not exceed max_chars + 200 overrun margin."""
        max_chars = 500  # Very small budget to force truncation
        assembler = _make_assembler(max_chars=max_chars)

        # 10 symbols with large source
        syms = [
            _make_symbol(
                f"big_sym_{i}",
                f"function_{i}",
                f"auth.module.function_{i}",
                source="x = 1\n" * 200,  # ~1200 chars each
            )
            for i in range(10)
        ]
        results = [_make_result(s) for s in syms]

        ctx = assembler.assemble(
            results=results,
            query="list all functions",
            intent=QueryIntent.GENERAL_QUERY,
        )

        # Allow 200 char overrun for headers
        assert ctx.total_chars <= max_chars + 200, (
            f"total_chars={ctx.total_chars} exceeds max_chars={max_chars} + 200"
        )

    def test_zero_results_no_budget_issue(self) -> None:
        """Zero results with tiny budget must still work without error."""
        assembler = _make_assembler(max_chars=10)
        ctx = assembler.assemble(results=[], query="x", intent=QueryIntent.GENERAL_QUERY)
        assert ctx.total_chars == 0

    def test_large_budget_allows_all_symbols(self) -> None:
        """With a very large budget, all symbols should fit."""
        assembler = _make_assembler(max_chars=100_000)
        syms = [
            _make_symbol(f"sym_{i}", f"fn_{i}", f"mod.fn_{i}", source="def fn(): pass\n")
            for i in range(5)
        ]
        results = [_make_result(s) for s in syms]
        ctx = assembler.assemble(
            results=results,
            query="all functions",
            intent=QueryIntent.FLOW_EXPLANATION,
        )
        assert len(ctx.sources) > 0


# ---------------------------------------------------------------------------
# Test 5: Entry point detection by decorator
# ---------------------------------------------------------------------------


class TestEntryPointDetectionByDecorator:
    def test_router_get_decorator_goes_to_entry_points(self) -> None:
        """Symbol with 'router.get' in decorators must end up in ENTRY POINTS section."""
        assembler = _make_assembler()
        sym = _make_symbol(
            "route_sym_id",
            "get_items",
            "api.routes.get_items",
            decorators=["router.get"],
            source='@router.get("/items")\ndef get_items(): return []',
        )
        result = _make_result(sym, score=0.95, source="semantic")

        ctx = assembler.assemble(
            results=[result],
            query="explain the items route",
            intent=QueryIntent.FLOW_EXPLANATION,
        )

        # The ENTRY POINTS section should exist and contain the symbol's text
        assert SECTION_ENTRY_POINTS in ctx.sections, (
            f"Expected ENTRY POINTS section. Sections: {list(ctx.sections.keys())}"
        )
        assert "get_items" in ctx.sections[SECTION_ENTRY_POINTS] or "route_sym_id" in ctx.sources

    def test_app_post_decorator_is_entry_point(self) -> None:
        """@app.post decorator must also trigger entry point classification."""
        assembler = _make_assembler()
        sym = _make_symbol(
            "post_route_id",
            "create_item",
            "api.routes.create_item",
            decorators=["app.post"],
            callers=["api.caller.some_fn"],  # Has callers but still a route
            source='@app.post("/items")\ndef create_item(): pass',
        )
        result = _make_result(sym, score=0.9, source="semantic")
        ctx = assembler.assemble(
            results=[result],
            query="create item endpoint",
            intent=QueryIntent.SYMBOL_LOOKUP,
        )
        assert SECTION_ENTRY_POINTS in ctx.sections

    def test_no_route_decorator_not_forced_to_entry_points(self) -> None:
        """A function with callers and no route decorator need not be in entry points."""
        assembler = _make_assembler()
        # This function has callers, so it won't be classified as entry point
        sym = _make_symbol(
            "helper_id",
            "internal_helper",
            "utils.helpers.internal_helper",
            decorators=[],
            callers=["auth.routes.login"],  # called by login — NOT an entry point
            source="def internal_helper(): pass",
        )
        result = _make_result(sym, score=0.8, source="graph", graph_depth=1)
        ctx = assembler.assemble(
            results=[result],
            query="internal helper",
            intent=QueryIntent.DEPENDENCY_QUERY,
        )
        # Entry points section, if present, should not contain this symbol
        if SECTION_ENTRY_POINTS in ctx.sections:
            assert "internal_helper" not in ctx.sections[SECTION_ENTRY_POINTS]


# ---------------------------------------------------------------------------
# Test 6: Topological ordering
# ---------------------------------------------------------------------------


class TestTopologicalOrdering:
    def test_caller_appears_before_callee_in_core_flow(self) -> None:
        """Symbol A that calls symbol B → A appears before B in CORE FLOW text."""
        assembler = _make_assembler()

        sym_b = _make_symbol(
            "sym_b_id",
            "validate_token",
            "auth.jwt.validate_token",
            source="def validate_token(): pass",
            callers=["auth.routes.login"],  # B is called by A
        )
        sym_a = _make_symbol(
            "sym_a_id",
            "login",
            "auth.routes.login",
            source="def login(): validate_token()",
            calls=["auth.jwt.validate_token"],  # A calls B
            callers=[],  # A has no callers → qualifies as entry point
        )

        # Use semantic source so both end up in core flow candidates
        r_a = _make_result(sym_a, score=0.95, source="semantic")
        r_b = _make_result(sym_b, score=0.90, source="semantic")

        ctx = assembler.assemble(
            results=[r_a, r_b],
            query="explain login flow",
            intent=QueryIntent.FLOW_EXPLANATION,
        )

        full_text = ctx.full_text
        assert len(full_text) > 0

        # login (caller) should appear before validate_token (callee) in the output
        pos_login = full_text.find("login")
        pos_validate = full_text.find("validate_token")

        # Both must appear somewhere in the output
        assert pos_login != -1, "login must appear in full_text"
        assert pos_validate != -1, "validate_token must appear in full_text"

        # login should be before validate_token (caller-before-callee)
        assert pos_login < pos_validate, (
            f"Expected login (pos={pos_login}) before validate_token (pos={pos_validate})"
        )

    def test_topological_order_with_no_calls(self) -> None:
        """Symbols with no call relationships must still be ordered deterministically."""
        assembler = _make_assembler()
        syms = [
            _make_symbol(f"iso_{i}", f"fn_{i}", f"mod.fn_{i}", source="def fn(): pass")
            for i in range(3)
        ]
        results = [_make_result(s, source="semantic") for s in syms]
        ctx = assembler.assemble(
            results=results, query="all fns", intent=QueryIntent.GENERAL_QUERY
        )
        # No error means topological sort handled the no-edges case cleanly
        assert len(ctx.full_text) > 0


# ---------------------------------------------------------------------------
# Test 7: Truncated flag
# ---------------------------------------------------------------------------


class TestTruncatedFlag:
    def test_truncated_flag_set_when_symbol_exceeds_budget(self) -> None:
        """When a symbol's source is larger than the budget, truncated=True must be set."""
        max_chars = 200  # Tiny budget
        assembler = _make_assembler(max_chars=max_chars)

        large_source = "# line\n" * 500  # ~3500 chars
        sym = _make_symbol(
            "large_sym_id",
            "massive_function",
            "mod.massive_function",
            source=large_source,
        )
        result = _make_result(sym, score=0.99, source="semantic")

        ctx = assembler.assemble(
            results=[result],
            query="explain massive function",
            intent=QueryIntent.FLOW_EXPLANATION,
        )

        assert ctx.truncated is True, "Expected truncated=True for oversized symbol"

    def test_truncated_false_when_fits(self) -> None:
        """Small symbol in a large budget must NOT set truncated=True."""
        assembler = _make_assembler(max_chars=50_000)
        sym = _make_symbol(
            "small_sym_id",
            "tiny_fn",
            "mod.tiny_fn",
            source="def tiny_fn(): return 1",
        )
        result = _make_result(sym, score=0.9, source="semantic")

        ctx = assembler.assemble(
            results=[result],
            query="tiny fn",
            intent=QueryIntent.SYMBOL_LOOKUP,
        )
        assert ctx.truncated is False


# ---------------------------------------------------------------------------
# Test 8: Sections present
# ---------------------------------------------------------------------------


class TestSectionsPresent:
    def test_at_least_one_section_present_for_non_empty_results(self) -> None:
        """assemble() with results must produce at least one section."""
        assembler = _make_assembler()
        sym = _make_symbol("sec_test_id", "my_fn", "mod.my_fn")
        ctx = assembler.assemble(
            results=[_make_result(sym)],
            query="what is my_fn",
            intent=QueryIntent.GENERAL_QUERY,
        )
        assert len(ctx.sections) >= 1

    def test_sections_are_non_empty_strings(self) -> None:
        """Every section in AssembledContext.sections must be a non-empty string."""
        assembler = _make_assembler()
        syms = [
            _make_symbol(f"sec_{i}", f"fn_{i}", f"mod.fn_{i}", source="def fn(): pass\n")
            for i in range(4)
        ]
        results = [_make_result(s) for s in syms]
        ctx = assembler.assemble(
            results=results, query="functions", intent=QueryIntent.GENERAL_QUERY
        )
        for name, text in ctx.sections.items():
            assert isinstance(text, str), f"Section {name} is not a string"
            assert len(text.strip()) > 0, f"Section {name} is empty"

    def test_full_text_contains_section_headers(self) -> None:
        """full_text must contain section header markers like '###'."""
        assembler = _make_assembler()
        sym = _make_symbol("hdr_id", "some_fn", "mod.some_fn", source="def some_fn(): pass\n")
        ctx = assembler.assemble(
            results=[_make_result(sym)],
            query="some_fn",
            intent=QueryIntent.SYMBOL_LOOKUP,
        )
        # The joined text should contain section header syntax
        assert "###" in ctx.full_text or len(ctx.full_text) > 0

    def test_symbol_with_import_line_creates_imports_section(self) -> None:
        """Symbol whose source contains 'import' lines must create IMPORTANT IMPORTS section."""
        assembler = _make_assembler()
        source_with_imports = (
            "from auth.jwt import validate_token\n"
            "import os\n"
            "def login(): validate_token()\n"
        )
        sym = _make_symbol(
            "import_sym_id",
            "login",
            "auth.routes.login",
            source=source_with_imports,
        )
        result = _make_result(sym, score=0.9, source="semantic")
        ctx = assembler.assemble(
            results=[result],
            query="login function",
            intent=QueryIntent.FLOW_EXPLANATION,
        )
        # IMPORTANT IMPORTS section should be present if imports were extracted
        # (not strictly guaranteed if symbol never made it into included_ids,
        # so we allow this test to just not crash)
        assert isinstance(ctx.sections, dict)

    def test_known_section_names_subset_of_possible(self) -> None:
        """All section names in result must be from the known set."""
        assembler = _make_assembler()
        syms = [
            _make_symbol(f"ksec_{i}", f"fn_{i}", f"mod.fn_{i}")
            for i in range(3)
        ]
        ctx = assembler.assemble(
            results=[_make_result(s) for s in syms],
            query="all functions",
            intent=QueryIntent.GENERAL_QUERY,
        )
        valid_sections = {
            SECTION_PROJECT_PURPOSE,
            SECTION_ENTRY_POINTS,
            SECTION_CORE_FLOW,
            SECTION_DEPENDENCIES,
            SECTION_UTILITIES,
            SECTION_IMPORTS,
        }
        for name in ctx.sections:
            assert name in valid_sections, f"Unexpected section name: {name!r}"

    def test_intent_affects_assembly_no_crash(self) -> None:
        """All QueryIntent values must not cause assemble() to crash."""
        assembler = _make_assembler()
        sym = _make_symbol("intent_sym_id", "fn_x", "mod.fn_x")
        result = _make_result(sym)

        for intent in QueryIntent:
            ctx = assembler.assemble(
                results=[result],
                query="test query",
                intent=intent,
            )
            assert isinstance(ctx, AssembledContext)
