from __future__ import annotations

"""
Tests for app.graph.graph_engine.CodeGraph

Strategy:
- Build ParsedFile objects manually for a small 3-file mock repository:
    File A  auth/routes.py  → function login() calls validate_token()
    File B  auth/jwt.py     → function validate_token(), class JWTHandler(BaseHandler)
    File C  auth/base.py    → class BaseHandler
- Build a real SymbolRegistry and a real CodeGraph
- Verify graph structure, traversal, impact analysis, serialisation

Node ID conventions (from graph_engine):
    file  → "file::{normalized_path}"
    func  → "func::{qualified_name}"
    class → "class::{qualified_name}"
"""

import json
from pathlib import Path

import pytest

from app.graph.graph_engine import CodeGraph, _func_node_id, _class_node_id, _file_node_id
from app.parsing.symbol_registry import build_registry
from app.schemas.models import (
    ParsedCall,
    ParsedClass,
    ParsedFile,
    ParsedFunction,
    ParsedImport,
)


# ---------------------------------------------------------------------------
# Repo constants
# ---------------------------------------------------------------------------

ROUTES_PATH = "/repo/auth/routes.py"
JWT_PATH = "/repo/auth/jwt.py"
BASE_PATH = "/repo/auth/base.py"

ROUTES_MODULE = "auth.routes"
JWT_MODULE = "auth.jwt"
BASE_MODULE = "auth.base"


# ---------------------------------------------------------------------------
# Helpers to create parsed objects
# ---------------------------------------------------------------------------


def _fn(
    name: str,
    module: str,
    file_path: str,
    *,
    class_name: str | None = None,
    calls: list[ParsedCall] | None = None,
    decorators: list[str] | None = None,
    line_start: int = 1,
    line_end: int = 10,
) -> ParsedFunction:
    qname = f"{module}.{class_name}.{name}" if class_name else f"{module}.{name}"
    return ParsedFunction(
        name=name,
        qualified_name=qname,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        source=f"def {name}(): pass",
        calls=calls or [],
        decorators=decorators or [],
        is_async=False,
        is_method=class_name is not None,
        class_name=class_name,
    )


def _cls(
    name: str,
    module: str,
    file_path: str,
    methods: list[ParsedFunction] | None = None,
    bases: list[str] | None = None,
) -> ParsedClass:
    return ParsedClass(
        name=name,
        qualified_name=f"{module}.{name}",
        file_path=file_path,
        line_start=1,
        line_end=30,
        source=f"class {name}: pass",
        bases=bases or [],
        methods=methods or [],
    )


# ---------------------------------------------------------------------------
# Fixture: the 3-file mock repo (shared across all tests in this module)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mock_graph() -> CodeGraph:
    """
    Build and return a CodeGraph for the 3-file mock repository.

    auth/routes.py:
        imports validate_token from auth.jwt
        function login() calls validate_token()

    auth/jwt.py:
        function validate_token()
        class JWTHandler(BaseHandler)

    auth/base.py:
        class BaseHandler
    """
    # --- auth/base.py ---
    base_handler_cls = _cls("BaseHandler", BASE_MODULE, BASE_PATH)
    pf_base = ParsedFile(
        path=BASE_PATH,
        module_name=BASE_MODULE,
        functions=[],
        classes=[base_handler_cls],
    )

    # --- auth/jwt.py ---
    validate_token_fn = _fn("validate_token", JWT_MODULE, JWT_PATH)
    jwt_handler_cls = _cls(
        "JWTHandler", JWT_MODULE, JWT_PATH,
        bases=["auth.base.BaseHandler"],
    )
    pf_jwt = ParsedFile(
        path=JWT_PATH,
        module_name=JWT_MODULE,
        imports=[],
        functions=[validate_token_fn],
        classes=[jwt_handler_cls],
    )

    # --- auth/routes.py ---
    # login() calls validate_token()
    call_to_validate = ParsedCall(name="validate_token", line=5)
    login_fn = _fn(
        "login", ROUTES_MODULE, ROUTES_PATH,
        calls=[call_to_validate],
    )
    imp_validate = ParsedImport(
        module="auth.jwt",
        names=["validate_token"],
        is_from=True,
        line=1,
    )
    pf_routes = ParsedFile(
        path=ROUTES_PATH,
        module_name=ROUTES_MODULE,
        imports=[imp_validate],
        functions=[login_fn],
        classes=[],
    )

    parsed_files = [pf_base, pf_jwt, pf_routes]
    registry = build_registry(parsed_files)

    graph = CodeGraph()
    graph.build(parsed_files, registry)

    return graph


# ---------------------------------------------------------------------------
# Helper to resolve expected node IDs
# ---------------------------------------------------------------------------


def nid_file(path: str) -> str:
    return _file_node_id(path)


def nid_func(qname: str) -> str:
    return _func_node_id(qname)


def nid_class(qname: str) -> str:
    return _class_node_id(qname)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNodesCreated:
    def test_file_nodes_exist(self, mock_graph: CodeGraph) -> None:
        """All 3 file nodes must be present in the graph."""
        for path in (ROUTES_PATH, JWT_PATH, BASE_PATH):
            node_id = nid_file(path)
            data = mock_graph.get_node_data(node_id)
            assert data is not None, f"File node missing: {node_id}"
            assert data["node_type"] == "file"

    def test_function_nodes_exist(self, mock_graph: CodeGraph) -> None:
        """login and validate_token function nodes must exist."""
        for qname in ("auth.routes.login", "auth.jwt.validate_token"):
            node_id = nid_func(qname)
            data = mock_graph.get_node_data(node_id)
            assert data is not None, f"Function node missing: {node_id}"
            assert data["node_type"] == "function"

    def test_class_nodes_exist(self, mock_graph: CodeGraph) -> None:
        """JWTHandler and BaseHandler class nodes must exist."""
        for qname in ("auth.jwt.JWTHandler", "auth.base.BaseHandler"):
            node_id = nid_class(qname)
            data = mock_graph.get_node_data(node_id)
            assert data is not None, f"Class node missing: {node_id}"
            assert data["node_type"] == "class"


class TestDefinesEdges:
    def test_file_defines_function(self, mock_graph: CodeGraph) -> None:
        """auth/routes.py file node must have DEFINES edge to login function."""
        file_id = nid_file(ROUTES_PATH)
        login_id = nid_func("auth.routes.login")

        # The edge should exist
        assert mock_graph.graph.has_edge(file_id, login_id), (
            f"Expected DEFINES edge from {file_id} to {login_id}"
        )
        edge_data = mock_graph.graph.edges[file_id, login_id]
        assert edge_data.get("edge_type") == "DEFINES"

    def test_file_defines_class(self, mock_graph: CodeGraph) -> None:
        """auth/jwt.py file node must have DEFINES edge to JWTHandler class."""
        file_id = nid_file(JWT_PATH)
        cls_id = nid_class("auth.jwt.JWTHandler")

        assert mock_graph.graph.has_edge(file_id, cls_id)
        edge_data = mock_graph.graph.edges[file_id, cls_id]
        assert edge_data.get("edge_type") == "DEFINES"

    def test_get_neighbors_defines(self, mock_graph: CodeGraph) -> None:
        """get_neighbors_by_edge_type with DEFINES must return defined symbols."""
        file_id = nid_file(JWT_PATH)
        defines_targets = mock_graph.get_neighbors_by_edge_type(file_id, "DEFINES")

        validate_id = nid_func("auth.jwt.validate_token")
        jwt_cls_id = nid_class("auth.jwt.JWTHandler")

        assert validate_id in defines_targets
        assert jwt_cls_id in defines_targets


class TestCallsEdge:
    def test_calls_edge_login_to_validate_token(self, mock_graph: CodeGraph) -> None:
        """login must have a CALLS edge to validate_token."""
        login_id = nid_func("auth.routes.login")
        validate_id = nid_func("auth.jwt.validate_token")

        assert mock_graph.graph.has_edge(login_id, validate_id), (
            "Expected CALLS edge from login to validate_token"
        )
        edge_data = mock_graph.graph.edges[login_id, validate_id]
        assert edge_data.get("edge_type") == "CALLS"

    def test_no_spurious_calls_edges(self, mock_graph: CodeGraph) -> None:
        """validate_token has no outgoing CALLS edges (it calls nothing)."""
        validate_id = nid_func("auth.jwt.validate_token")
        outgoing_calls = mock_graph.get_neighbors_by_edge_type(validate_id, "CALLS")
        assert outgoing_calls == []


class TestInheritsEdge:
    def test_inherits_edge_jwt_handler_to_base_handler(self, mock_graph: CodeGraph) -> None:
        """JWTHandler must have an INHERITS edge to BaseHandler."""
        jwt_cls_id = nid_class("auth.jwt.JWTHandler")
        base_cls_id = nid_class("auth.base.BaseHandler")

        assert mock_graph.graph.has_edge(jwt_cls_id, base_cls_id), (
            "Expected INHERITS edge from JWTHandler to BaseHandler"
        )
        edge_data = mock_graph.graph.edges[jwt_cls_id, base_cls_id]
        assert edge_data.get("edge_type") == "INHERITS"

    def test_base_handler_no_inherits_edges(self, mock_graph: CodeGraph) -> None:
        """BaseHandler (root class) must have no outgoing INHERITS edges."""
        base_cls_id = nid_class("auth.base.BaseHandler")
        inherits = mock_graph.get_neighbors_by_edge_type(base_cls_id, "INHERITS")
        assert inherits == []


class TestGetDependencies:
    def test_get_dependencies_for_login(self, mock_graph: CodeGraph) -> None:
        """get_dependencies for login must include validate_token."""
        login_id = nid_func("auth.routes.login")
        deps = mock_graph.get_dependencies(login_id, depth=2)

        validate_id = nid_func("auth.jwt.validate_token")
        assert validate_id in deps

    def test_get_dependencies_empty_for_leaf(self, mock_graph: CodeGraph) -> None:
        """validate_token depends on nothing → get_dependencies returns []."""
        validate_id = nid_func("auth.jwt.validate_token")
        deps = mock_graph.get_dependencies(validate_id, depth=2)
        # Should be empty (no outgoing CALLS/IMPORTS)
        assert validate_id not in deps

    def test_get_dependencies_unknown_node(self, mock_graph: CodeGraph) -> None:
        """Unknown node_id must return empty list without error."""
        result = mock_graph.get_dependencies("func::does.not.exist", depth=2)
        assert result == []


class TestGetDependents:
    def test_get_dependents_for_validate_token(self, mock_graph: CodeGraph) -> None:
        """get_dependents for validate_token must include login."""
        validate_id = nid_func("auth.jwt.validate_token")
        dependents = mock_graph.get_dependents(validate_id, depth=2)

        login_id = nid_func("auth.routes.login")
        assert login_id in dependents

    def test_get_dependents_empty_for_root(self, mock_graph: CodeGraph) -> None:
        """login has no callers → get_dependents must return []."""
        login_id = nid_func("auth.routes.login")
        dependents = mock_graph.get_dependents(login_id, depth=2)
        # login is called by nobody
        assert login_id not in dependents

    def test_get_dependents_unknown_node(self, mock_graph: CodeGraph) -> None:
        """Unknown node returns empty list."""
        result = mock_graph.get_dependents("func::not.there", depth=2)
        assert result == []


class TestGetImpact:
    def test_impact_includes_login_in_directly_affected(self, mock_graph: CodeGraph) -> None:
        """
        Changing validate_token directly affects login (its direct caller).
        ImpactAnalysisResult.directly_affected must contain login's node_id.
        """
        validate_id = nid_func("auth.jwt.validate_token")
        login_id = nid_func("auth.routes.login")

        impact = mock_graph.get_impact(validate_id)
        assert login_id in impact.directly_affected

    def test_impact_target_symbol_correct(self, mock_graph: CodeGraph) -> None:
        """ImpactAnalysisResult.target_symbol must match the node's qualified_name."""
        validate_id = nid_func("auth.jwt.validate_token")
        impact = mock_graph.get_impact(validate_id)
        assert impact.target_symbol == "auth.jwt.validate_token"

    def test_impact_nonexistent_node(self, mock_graph: CodeGraph) -> None:
        """Nonexistent node must return an ImpactAnalysisResult with no affected symbols."""
        impact = mock_graph.get_impact("func::does.not.exist")
        assert impact.directly_affected == []
        assert impact.transitively_affected == []
        assert impact.impact_depth == 0

    def test_impact_depth_is_positive(self, mock_graph: CodeGraph) -> None:
        """Impact depth for validate_token must be >= 1 (login is depth 1)."""
        validate_id = nid_func("auth.jwt.validate_token")
        impact = mock_graph.get_impact(validate_id)
        assert impact.impact_depth >= 1


class TestFindSymbolNodes:
    def test_find_symbol_nodes_validate(self, mock_graph: CodeGraph) -> None:
        """find_symbol_nodes('validate') must return validate_token's node_id."""
        results = mock_graph.find_symbol_nodes("validate")
        validate_id = nid_func("auth.jwt.validate_token")
        assert validate_id in results

    def test_find_symbol_nodes_base(self, mock_graph: CodeGraph) -> None:
        """find_symbol_nodes('base') must include BaseHandler class node."""
        results = mock_graph.find_symbol_nodes("base")
        base_id = nid_class("auth.base.BaseHandler")
        assert base_id in results

    def test_find_symbol_nodes_case_insensitive(self, mock_graph: CodeGraph) -> None:
        """find_symbol_nodes must be case-insensitive."""
        results_lower = mock_graph.find_symbol_nodes("jwt")
        results_upper = mock_graph.find_symbol_nodes("JWT")
        assert set(results_lower) == set(results_upper)

    def test_find_symbol_nodes_no_match(self, mock_graph: CodeGraph) -> None:
        """find_symbol_nodes with no match must return []."""
        results = mock_graph.find_symbol_nodes("xyz_totally_unknown_qwerty")
        assert results == []


class TestSaveLoadRoundtrip:
    def test_save_load_roundtrip(self, mock_graph: CodeGraph, tmp_path: Path) -> None:
        """Save graph to file, load it back, verify node/edge counts match."""
        save_path = tmp_path / "graph.json"
        mock_graph.save(save_path)

        assert save_path.exists()
        assert save_path.stat().st_size > 0

        loaded = CodeGraph.load(save_path)

        assert loaded.node_count() == mock_graph.node_count(), (
            f"Node count mismatch: {loaded.node_count()} vs {mock_graph.node_count()}"
        )
        assert loaded.edge_count() == mock_graph.edge_count(), (
            f"Edge count mismatch: {loaded.edge_count()} vs {mock_graph.edge_count()}"
        )

    def test_save_produces_valid_json(self, mock_graph: CodeGraph, tmp_path: Path) -> None:
        """Saved graph file must be valid JSON."""
        save_path = tmp_path / "graph_valid.json"
        mock_graph.save(save_path)
        with save_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert "graph" in data

    def test_loaded_graph_preserves_node_ids(self, mock_graph: CodeGraph, tmp_path: Path) -> None:
        """After load, get_node_id must still resolve qualified names."""
        save_path = tmp_path / "graph_ids.json"
        mock_graph.save(save_path)
        loaded = CodeGraph.load(save_path)

        validate_id = loaded.get_node_id("auth.jwt.validate_token")
        assert validate_id == nid_func("auth.jwt.validate_token")

    def test_loaded_graph_edges_intact(self, mock_graph: CodeGraph, tmp_path: Path) -> None:
        """After load, CALLS edge from login to validate_token must still exist."""
        save_path = tmp_path / "graph_edges.json"
        mock_graph.save(save_path)
        loaded = CodeGraph.load(save_path)

        login_id = nid_func("auth.routes.login")
        validate_id = nid_func("auth.jwt.validate_token")
        assert loaded.graph.has_edge(login_id, validate_id)


class TestFindEntryPoints:
    def test_login_is_entry_point(self, mock_graph: CodeGraph) -> None:
        """login() is never called by another function — it must be an entry point."""
        login_id = nid_func("auth.routes.login")
        all_func_nodes = [
            nid for nid, data in mock_graph.graph.nodes(data=True)
            if data.get("node_type") == "function"
        ]
        entry_points = mock_graph.find_entry_points(all_func_nodes)
        assert login_id in entry_points

    def test_validate_token_not_entry_point(self, mock_graph: CodeGraph) -> None:
        """validate_token is called by login — it must NOT be an entry point."""
        validate_id = nid_func("auth.jwt.validate_token")
        all_func_nodes = [
            nid for nid, data in mock_graph.graph.nodes(data=True)
            if data.get("node_type") == "function"
        ]
        entry_points = mock_graph.find_entry_points(all_func_nodes)
        assert validate_id not in entry_points

    def test_find_entry_points_empty_input(self, mock_graph: CodeGraph) -> None:
        """Empty input must return empty list."""
        assert mock_graph.find_entry_points([]) == []

    def test_find_entry_points_unknown_nodes(self, mock_graph: CodeGraph) -> None:
        """Unknown node ids must be silently skipped."""
        result = mock_graph.find_entry_points(["func::does.not.exist"])
        assert result == []
