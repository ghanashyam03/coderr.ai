from __future__ import annotations

"""
Integration Tests for Coderr Architecture and Execution Intelligence.

Verifies:
1. Architectural analysis (coupling instability, circular imports, hubs, oversized orchestrators, dead code).
2. Execution flow reconstruction (BFS pathways, steps, roles).
3. Utility helper detection and signature-only source compression.
"""

from unittest.mock import MagicMock
from pathlib import Path
import pytest

from app.graph.graph_engine import CodeGraph, _func_node_id, _class_node_id, _file_node_id
from app.graph.flow_reconstructor import ExecutionFlowReconstructor
from app.reasoning.context_assembler import ContextAssembler
from app.parsing.symbol_registry import build_registry
from app.schemas.models import (
    ParsedCall,
    ParsedClass,
    ParsedFile,
    ParsedFunction,
    ParsedImport,
    RetrievalResult,
    CodeSymbol,
    SymbolType,
    QueryIntent,
)


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
    source: str = "",
    docstring: str | None = None,
) -> ParsedFunction:
    qname = f"{module}.{class_name}.{name}" if class_name else f"{module}.{name}"
    return ParsedFunction(
        name=name,
        qualified_name=qname,
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        source=source or f"def {name}(): pass",
        docstring=docstring,
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
    source: str = "",
) -> ParsedClass:
    return ParsedClass(
        name=name,
        qualified_name=f"{module}.{name}",
        file_path=file_path,
        line_start=1,
        line_end=30,
        source=source or f"class {name}: pass",
        bases=bases or [],
        methods=methods or [],
    )


@pytest.fixture
def sample_arch_repo() -> list[ParsedFile]:
    """
    Creates a mock 4-file repository setup:
    1. app/routes.py:
       Imports from app/orchestrator.py.
       Function `login_route` decorated with @app.post. Calls `Orchestrator.login`.
    2. app/orchestrator.py:
       Imports from app/services.py.
       Class `Orchestrator` with method `login` which calls `authenticate_user`.
    3. app/services.py:
       Imports from app/utils.py.
       Function `authenticate_user` calls `db_lookup` and `log_info`.
    4. app/utils.py:
       Function `db_lookup` (calls no one).
       Function `log_info` (calls no one).
       Function `dead_utility` (calls no one, never called).
    
    We also construct a circular import between app/routes.py and app/orchestrator.py for testing:
    routes.py imports orchestrator.py, and orchestrator.py imports routes.py.
    """
    p1 = "/repo/app/routes.py"
    p2 = "/repo/app/orchestrator.py"
    p3 = "/repo/app/services.py"
    p4 = "/repo/app/utils.py"

    m1 = "app.routes"
    m2 = "app.orchestrator"
    m3 = "app.services"
    m4 = "app.utils"

    f1 = ParsedFile(
        path=p1,
        module_name=m1,
        imports=[
            ParsedImport(module=m2, names=["Orchestrator"], line=2, is_from=True),
        ],
        functions=[
            _fn("login_route", m1, p1, decorators=["app.post"], calls=[
                ParsedCall(name="Orchestrator.login", line=4, resolved="app.orchestrator.Orchestrator.login")
            ])
        ]
    )

    f2 = ParsedFile(
        path=p2,
        module_name=m2,
        imports=[
            ParsedImport(module=m1, names=["login_route"], line=2, is_from=True), # circular import
            ParsedImport(module=m3, names=["authenticate_user"], line=3, is_from=True),
        ],
        classes=[
            _cls("Orchestrator", m2, p2, methods=[
                _fn("login", m2, p2, class_name="Orchestrator", calls=[
                    ParsedCall(name="authenticate_user", line=6, resolved="app.services.authenticate_user")
                ])
            ])
        ]
    )

    f3 = ParsedFile(
        path=p3,
        module_name=m3,
        imports=[
            ParsedImport(module=m4, names=["db_lookup", "log_info"], line=2, is_from=True)
        ],
        functions=[
            _fn("authenticate_user", m3, p3, calls=[
                ParsedCall(name="db_lookup", line=5, resolved="app.utils.db_lookup"),
                ParsedCall(name="log_info", line=6, resolved="app.utils.log_info"),
            ])
        ]
    )

    f4 = ParsedFile(
        path=p4,
        module_name=m4,
        functions=[
            _fn("db_lookup", m4, p4),
            _fn("log_info", m4, p4),
            _fn("dead_utility", m4, p4), # dead code
        ]
    )

    return [f1, f2, f3, f4]


def test_architectural_cycles_and_coupling(sample_arch_repo) -> None:
    """Validate cycle detection and coupling metrics calculation."""
    registry = build_registry(sample_arch_repo)
    graph = CodeGraph()
    graph.build(sample_arch_repo, registry)

    analysis = graph.analyze_architecture()

    # Verify Circular Imports
    assert len(analysis["circular_imports"]) > 0
    assert any("routes.py" in cycle[0] or "orchestrator.py" in cycle[0] for cycle in analysis["circular_imports"])

    # Verify Coupling
    coupling = analysis["coupling_instability"]
    assert "/repo/app/routes.py" in coupling
    assert "/repo/app/utils.py" in coupling

    routes_metrics = coupling["/repo/app/routes.py"]
    utils_metrics = coupling["/repo/app/utils.py"]

    # routes has out-coupling (imports orchestrator) and in-coupling (imported by orchestrator)
    assert routes_metrics["efferent_coupling"] == 1
    assert routes_metrics["afferent_coupling"] == 1
    assert routes_metrics["instability"] == 0.5

    # utils has 0 out-coupling, only in-coupling from services
    assert utils_metrics["efferent_coupling"] == 0
    assert utils_metrics["afferent_coupling"] == 1
    assert utils_metrics["instability"] == 0.0


def test_architectural_hubs_orchestrators_and_dead_code(sample_arch_repo) -> None:
    """Validate bedrock hubs, orchestrators, and dead code candidates."""
    registry = build_registry(sample_arch_repo)
    graph = CodeGraph()
    graph.build(sample_arch_repo, registry)

    analysis = graph.analyze_architecture()

    # authenticate_user has in-degree of 1 (from Orchestrator.login), but db_lookup/log_info both have 1 as well
    # Let's verify dead code detection
    dead = [d["qualified_name"] for d in analysis["dead_code_candidates"]]
    assert "app.utils.dead_utility" in dead
    assert "app.routes.login_route" not in dead # login_route is a decorated route


def test_execution_flow_reconstruction(sample_arch_repo) -> None:
    """Validate ExecutionFlowReconstructor computes proximity and BFS step roles correctly."""
    registry = build_registry(sample_arch_repo)
    graph = CodeGraph()
    graph.build(sample_arch_repo, registry)

    reconstructor = ExecutionFlowReconstructor(graph)
    entrypoints = reconstructor.find_all_entrypoints()

    # login_route has decorator @app.post, so it is a likely entrypoint
    login_node_id = _func_node_id("app.routes.login_route")
    assert login_node_id in entrypoints

    # Trace flow from login_route
    flow = reconstructor.trace_execution_flow(login_node_id, max_depth=4)
    steps = flow["steps"]

    # Expect: login_route -> Orchestrator.login -> authenticate_user -> [db_lookup, log_info]
    assert len(steps) >= 4
    
    # Check depth and roles
    assert steps[0]["qualified_name"] == "app.routes.login_route"
    assert steps[0]["role"] == "entrypoint"
    assert steps[0]["depth"] == 0

    assert steps[1]["qualified_name"] == "app.orchestrator.Orchestrator.login"
    assert steps[1]["role"] == "orchestrator"
    assert steps[1]["depth"] == 1

    assert steps[2]["qualified_name"] == "app.services.authenticate_user"
    assert steps[2]["role"] == "service"
    assert steps[2]["depth"] == 2

    # Utilities are at depth 3
    utilities = [s["qualified_name"] for s in steps[3:]]
    assert "app.utils.db_lookup" in utilities
    assert "app.utils.log_info" in utilities


def test_compression_and_helper_rendering() -> None:
    """Validate ContextAssembler helper detection and signature-only source compression."""
    graph_mock = MagicMock()
    graph_mock.get_node_id.return_value = "func::app.utils.db_lookup"
    
    # Setup graph out_degree and in_degree for helper node
    di_graph = MagicMock()
    di_graph.out_degree.return_value = 0
    di_graph.in_degree.return_value = 1
    graph_mock.graph = di_graph

    assembler = ContextAssembler(graph_mock)

    # 1. Helper function
    sym = CodeSymbol(
        symbol_id="app/utils.py::db_lookup",
        symbol_type=SymbolType.FUNCTION,
        name="db_lookup",
        qualified_name="app.utils.db_lookup",
        file_path="app/utils.py",
        line_start=5,
        line_end=15,
        source="def db_lookup(query: str) -> dict:\n    \"\"\"Perform DB lookup.\"\"\"\n    res = conn.execute(query)\n    return res.fetchall()",
        docstring="Perform DB lookup.",
        repo_name="testrepo",
    )

    assert assembler._is_low_value_helper(sym) is True

    # 2. Source compression test
    compressed = assembler._compress_source(sym.source, sym.docstring)
    assert "def db_lookup(query: str) -> dict:" in compressed
    assert '"""\n    Perform DB lookup.\n    """' in compressed
    assert "res = conn.execute(query)" not in compressed
    assert "body omitted" in compressed


def test_calls_resolution_type_and_edge_annotation(sample_arch_repo) -> None:
    """Validate that resolution types are correctly mapped and CALLS edges are annotated in graph."""
    registry = build_registry(sample_arch_repo)
    
    # Check that raw AST calls resolved via imports are classified as IMPORT resolution type
    routes_file = [f for f in sample_arch_repo if "routes.py" in f.path][0]
    login_route_fn = routes_file.functions[0]
    call = login_route_fn.calls[0]
    assert call.resolved == "app.orchestrator.Orchestrator.login"
    assert call.resolution_type.value == "import"

    # Verify edge annotation in graph
    graph = CodeGraph()
    graph.build(sample_arch_repo, registry)
    
    caller_node = _func_node_id("app.routes.login_route")
    callee_node = _func_node_id("app.orchestrator.Orchestrator.login")
    assert graph.graph.has_edge(caller_node, callee_node)
    
    edge_data = graph.graph.edges[caller_node, callee_node]
    assert edge_data.get("resolution_type") == "import"


def test_path_confidence_scoring_and_helper_suppression(sample_arch_repo) -> None:
    """Validate cumulative confidence calculation and aggressive helper suppression."""
    registry = build_registry(sample_arch_repo)
    
    # 1. Update one call to HEURISTIC type to verify scoring decay
    orchestrator_file = [f for f in sample_arch_repo if "orchestrator.py" in f.path][0]
    login_method = orchestrator_file.classes[0].methods[0]
    login_method.calls[0].resolution_type = "heuristic" # mock heuristic resolution
    
    graph = CodeGraph()
    graph.build(sample_arch_repo, registry)

    reconstructor = ExecutionFlowReconstructor(graph)
    login_route_id = _func_node_id("app.routes.login_route")
    
    flow = reconstructor.trace_execution_flow(login_route_id, max_depth=3)
    steps = flow["steps"]
    
    # Expect: 
    # login_route (depth 0, conf 1.0, direct/import)
    # Orchestrator.login (depth 1, conf 1.0 * 1.0 = 1.0)
    # authenticate_user (depth 2, conf 1.0 * 0.7 = 0.7 via heuristic edge)
    # db_lookup (depth 3, conf 0.7 * 1.0 = 0.7 via direct edge)
    assert steps[0]["confidence"] == 1.0
    assert steps[1]["confidence"] == 1.0
    assert steps[2]["confidence"] == 0.7
    assert steps[3]["confidence"] == 0.7

    # 2. Verify Aggressive Suppress / Omit from Context Assembler
    assembler = ContextAssembler(graph)
    
    # Create CodeSymbol representations
    sym_dead = CodeSymbol(
        symbol_id="/repo/app/utils.py::dead_utility",
        symbol_type=SymbolType.FUNCTION,
        name="dead_utility",
        qualified_name="app.utils.dead_utility",
        file_path="/repo/app/utils.py",
        line_start=15,
        line_end=20,
        source="def dead_utility(): pass",
        repo_name="testrepo",
    )
    
    # Since dead_utility is isolated leaf utility (in-degree 0, out-degree 0), it is a helper leaf
    assert assembler._is_low_value_helper(sym_dead) is True
    
    # Run classification
    results = [RetrievalResult(symbol=sym_dead, score=0.8, source="semantic")]
    classified = assembler._classify_into_bins(results, QueryIntent.GENERAL_QUERY, None)
    
    # Expect dead_utility to be completely omitted/suppressed from context sections
    symbols = classified["symbols"]
    assert sym_dead not in symbols["IMPLEMENTATION DETAILS"]

