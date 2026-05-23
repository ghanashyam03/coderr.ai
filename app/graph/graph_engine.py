"""
app/graph/graph_engine.py
--------------------------
CodeGraph — the central graph engine for Coderr.

Stores a networkx DiGraph of file / class / function nodes connected by
DEFINES, IMPORTS, CALLS, and INHERITS edges.  Provides traversal helpers,
impact analysis, API-route detection, and JSON serialisation.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import networkx as nx

from app.schemas.models import ImpactAnalysisResult, ParsedFile

if TYPE_CHECKING:
    from app.parsing.symbol_registry import SymbolRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_API_DECORATOR_PATTERNS: tuple[str, ...] = (
    "route",
    ".get",
    ".post",
    ".put",
    ".delete",
    ".patch",
    "app.get",
    "app.post",
    "app.put",
    "app.delete",
    "app.patch",
    "router.get",
    "router.post",
    "router.put",
    "router.delete",
    "router.patch",
)

_CALLS_EDGE = "CALLS"
_DEFINES_EDGE = "DEFINES"
_IMPORTS_EDGE = "IMPORTS"
_INHERITS_EDGE = "INHERITS"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _normalize_path(path: str) -> str:
    """Return a lowercase, forward-slash-only representation of *path*."""
    return path.replace("\\", "/").lower()


def _file_node_id(path: str) -> str:
    return f"file::{_normalize_path(path)}"


def _class_node_id(qualified_name: str) -> str:
    return f"class::{qualified_name}"


def _func_node_id(qualified_name: str) -> str:
    return f"func::{qualified_name}"


def _is_api_route_node(data: dict[str, Any]) -> bool:
    """Return True when *data* (node attribute dict) looks like an API route."""
    decorators: list[str] = data.get("decorators", [])
    for dec in decorators:
        dec_lower = dec.lower()
        for pattern in _API_DECORATOR_PATTERNS:
            if pattern in dec_lower:
                return True
    return False


# ---------------------------------------------------------------------------
# CodeGraph
# ---------------------------------------------------------------------------


class CodeGraph:
    """
    In-memory directed graph over a parsed Python codebase.

    Node ID conventions
    -------------------
    * file node  →  ``file::{normalized_path}``
    * class node →  ``class::{qualified_name}``
    * func node  →  ``func::{qualified_name}``
    """

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        # qualified_name → node_id
        self._node_by_qualified: dict[str, str] = {}
        # normalized file_path → node_id
        self._file_to_node: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_node(self, node_id: str, **attrs: Any) -> None:
        """Add or update a node, keeping existing attributes on collision."""
        if node_id in self.graph:
            self.graph.nodes[node_id].update(attrs)
        else:
            self.graph.add_node(node_id, **attrs)

    def _add_edge(self, src: str, dst: str, edge_type: str, **attrs: Any) -> None:
        """Add a directed edge with an *edge_type* attribute and other attributes."""
        if not self.graph.has_node(src) or not self.graph.has_node(dst):
            return
        self.graph.add_edge(src, dst, edge_type=edge_type, **attrs)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, parsed_files: list[ParsedFile], registry: "SymbolRegistry") -> None:
        """
        Populate the graph from *parsed_files*.

        Phase order
        -----------
        1. Create all file / class / function nodes.
        2. Add DEFINES edges (file→class, file→function, class→method).
        3. Add IMPORTS edges (file→file, when import resolves inside repo).
        4. Add CALLS edges (function→function, resolved calls only).
        5. Add INHERITS edges (class→class via base class name lookup).
        6. Annotate *callers* attribute on target nodes from CALLS edges.
        """
        self.graph.clear()
        self._node_by_qualified.clear()
        self._file_to_node.clear()

        logger.info("Building CodeGraph from %d parsed files …", len(parsed_files))

        # Build a set of all known file paths (normalised) for IMPORTS resolution.
        known_file_paths: set[str] = {
            _normalize_path(pf.path) for pf in parsed_files if not pf.parse_error
        }

        # ----------------------------------------------------------------
        # Phase 1 & 2: Create nodes and DEFINES edges
        # ----------------------------------------------------------------
        for pf in parsed_files:
            if pf.parse_error:
                logger.debug("Skipping %s — parse error: %s", pf.path, pf.parse_error)
                continue

            file_id = _file_node_id(pf.path)
            norm_path = _normalize_path(pf.path)

            self._add_node(
                file_id,
                node_type="file",
                name=Path(pf.path).name,
                qualified_name=pf.module_name,
                file_path=pf.path,
                line_start=1,
                symbol_type="file",
                decorators=[],
                is_async=False,
            )
            self._node_by_qualified[pf.module_name] = file_id
            self._file_to_node[norm_path] = file_id

            # Top-level functions
            for fn in pf.functions:
                fn_id = _func_node_id(fn.qualified_name)
                self._add_node(
                    fn_id,
                    node_type="function",
                    name=fn.name,
                    qualified_name=fn.qualified_name,
                    file_path=fn.file_path,
                    line_start=fn.line_start,
                    symbol_type="function",
                    decorators=fn.decorators,
                    is_async=fn.is_async,
                )
                self._node_by_qualified[fn.qualified_name] = fn_id
                self.graph.add_edge(file_id, fn_id, edge_type=_DEFINES_EDGE)

            # Classes and their methods
            for cls in pf.classes:
                cls_id = _class_node_id(cls.qualified_name)
                self._add_node(
                    cls_id,
                    node_type="class",
                    name=cls.name,
                    qualified_name=cls.qualified_name,
                    file_path=cls.file_path,
                    line_start=cls.line_start,
                    symbol_type="class",
                    decorators=cls.decorators,
                    is_async=False,
                )
                self._node_by_qualified[cls.qualified_name] = cls_id
                self.graph.add_edge(file_id, cls_id, edge_type=_DEFINES_EDGE)

                for method in cls.methods:
                    m_id = _func_node_id(method.qualified_name)
                    self._add_node(
                        m_id,
                        node_type="function",
                        name=method.name,
                        qualified_name=method.qualified_name,
                        file_path=method.file_path,
                        line_start=method.line_start,
                        symbol_type="method",
                        decorators=method.decorators,
                        is_async=method.is_async,
                    )
                    self._node_by_qualified[method.qualified_name] = m_id
                    self.graph.add_edge(cls_id, m_id, edge_type=_DEFINES_EDGE)

        logger.info(
            "Phase 1+2 complete: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

        # ----------------------------------------------------------------
        # Phase 3: IMPORTS edges
        # ----------------------------------------------------------------
        for pf in parsed_files:
            if pf.parse_error:
                continue
            src_file_id = _file_node_id(pf.path)

            for imp in pf.imports:
                # Convert module path to potential file path candidates.
                # e.g. "auth.jwt" → "auth/jwt.py"  and  "auth/jwt/__init__.py"
                module_path = imp.module.replace(".", "/") if imp.module else None
                if module_path is None:
                    continue

                candidates = [
                    f"{module_path}.py",
                    f"{module_path}/__init__.py",
                ]
                for candidate in candidates:
                    candidate_norm = candidate.lower()
                    # Check if any known path ends with this candidate
                    target_file_id: Optional[str] = None
                    for known in known_file_paths:
                        if known.endswith(candidate_norm) or known == candidate_norm:
                            target_file_id = self._file_to_node.get(known)
                            break
                    if target_file_id and target_file_id != src_file_id:
                        self._add_edge(src_file_id, target_file_id, _IMPORTS_EDGE)
                        break

        logger.info(
            "Phase 3 complete: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

        # ----------------------------------------------------------------
        # Phase 4: CALLS edges
        # ----------------------------------------------------------------
        for pf in parsed_files:
            if pf.parse_error:
                continue

            all_functions = list(pf.functions)
            for cls in pf.classes:
                all_functions.extend(cls.methods)

            for fn in all_functions:
                caller_id = _func_node_id(fn.qualified_name)
                if not self.graph.has_node(caller_id):
                    continue
                for call in fn.calls:
                    if not call.resolved:
                        continue
                    callee_id = self._node_by_qualified.get(call.resolved)
                    if callee_id and callee_id != caller_id:
                        res_type = getattr(call, "resolution_type", "unresolved")
                        res_val = res_type.value if hasattr(res_type, "value") else str(res_type)
                        self._add_edge(caller_id, callee_id, _CALLS_EDGE, resolution_type=res_val)

        logger.info(
            "Phase 4 complete: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

        # ----------------------------------------------------------------
        # Phase 5: INHERITS edges
        # ----------------------------------------------------------------
        for pf in parsed_files:
            if pf.parse_error:
                continue
            for cls in pf.classes:
                cls_id = _class_node_id(cls.qualified_name)
                if not self.graph.has_node(cls_id):
                    continue
                for base_name in cls.bases:
                    # Attempt resolution via registry first.
                    resolved_base: Optional[str] = None
                    if hasattr(registry, "resolve_name"):
                        resolved_base = registry.resolve_name(base_name, pf.module_name)
                    # Fall back to direct lookup.
                    if not resolved_base:
                        resolved_base = base_name
                    base_id = self._node_by_qualified.get(resolved_base)
                    if base_id and base_id != cls_id:
                        self._add_edge(cls_id, base_id, _INHERITS_EDGE)

        logger.info(
            "Phase 5 complete: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

        # ----------------------------------------------------------------
        # Phase 6: Annotate callers on target nodes
        # ----------------------------------------------------------------
        for node_id in self.graph.nodes:
            self.graph.nodes[node_id].setdefault("callers", [])

        for src, dst, data in self.graph.edges(data=True):
            if data.get("edge_type") == _CALLS_EDGE:
                callers_list: list[str] = self.graph.nodes[dst].get("callers", [])
                if src not in callers_list:
                    callers_list.append(src)
                self.graph.nodes[dst]["callers"] = callers_list

        # Count edges by type for summary log.
        edge_type_counts: dict[str, int] = {}
        for _, _, edata in self.graph.edges(data=True):
            etype = edata.get("edge_type", "UNKNOWN")
            edge_type_counts[etype] = edge_type_counts.get(etype, 0) + 1

        logger.info(
            "CodeGraph build complete — nodes=%d  edges=%d  by_type=%s",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
            edge_type_counts,
        )

    # ------------------------------------------------------------------
    # Node look-ups
    # ------------------------------------------------------------------

    def get_node_id(self, qualified_name: str) -> Optional[str]:
        """Return the node_id for *qualified_name*, or ``None``."""
        return self._node_by_qualified.get(qualified_name)

    def get_node_data(self, node_id: str) -> Optional[dict[str, Any]]:
        """Return the attribute dict for *node_id*, or ``None``."""
        if node_id not in self.graph:
            return None
        return dict(self.graph.nodes[node_id])

    # ------------------------------------------------------------------
    # Traversal helpers
    # ------------------------------------------------------------------

    def get_dependencies(self, node_id: str, depth: int = 2) -> list[str]:
        """
        BFS forward along CALLS and IMPORTS edges.

        Returns all reachable node_ids up to *depth* hops away
        (not including *node_id* itself).
        """
        if node_id not in self.graph:
            return []

        visited: set[str] = {node_id}
        result: list[str] = []
        frontier: deque[tuple[str, int]] = deque([(node_id, 0)])

        while frontier:
            current, d = frontier.popleft()
            if d >= depth:
                continue
            for neighbor in self.graph.successors(current):
                edge_data = self.graph.edges[current, neighbor]
                if edge_data.get("edge_type") not in (_CALLS_EDGE, _IMPORTS_EDGE):
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    result.append(neighbor)
                    frontier.append((neighbor, d + 1))

        return result

    def get_dependents(self, node_id: str, depth: int = 2) -> list[str]:
        """
        BFS reverse along CALLS and IMPORTS edges.

        Returns all node_ids that (transitively) depend on *node_id*
        up to *depth* hops, excluding *node_id* itself.
        """
        if node_id not in self.graph:
            return []

        visited: set[str] = {node_id}
        result: list[str] = []
        frontier: deque[tuple[str, int]] = deque([(node_id, 0)])

        while frontier:
            current, d = frontier.popleft()
            if d >= depth:
                continue
            for predecessor in self.graph.predecessors(current):
                edge_data = self.graph.edges[predecessor, current]
                if edge_data.get("edge_type") not in (_CALLS_EDGE, _IMPORTS_EDGE):
                    continue
                if predecessor not in visited:
                    visited.add(predecessor)
                    result.append(predecessor)
                    frontier.append((predecessor, d + 1))

        return result

    def get_call_chain(self, node_id: str, depth: int = 3) -> list[str]:
        """
        DFS following CALLS edges only.

        Returns an ordered list of node_ids (excluding *node_id* itself).
        Cycle-safe: each node is visited at most once.
        """
        if node_id not in self.graph:
            return []

        result: list[str] = []
        visited: set[str] = {node_id}

        def _dfs(current: str, remaining: int) -> None:
            if remaining == 0:
                return
            for neighbor in self.graph.successors(current):
                edge_data = self.graph.edges[current, neighbor]
                if edge_data.get("edge_type") != _CALLS_EDGE:
                    continue
                if neighbor not in visited:
                    visited.add(neighbor)
                    result.append(neighbor)
                    _dfs(neighbor, remaining - 1)

        _dfs(node_id, depth)
        return result

    # ------------------------------------------------------------------
    # Impact analysis
    # ------------------------------------------------------------------

    def get_impact(self, node_id: str) -> ImpactAnalysisResult:
        """
        Reverse BFS to find everything that will be impacted if *node_id* changes.

        Returns an :class:`ImpactAnalysisResult` with:

        * ``directly_affected``     — predecessors at depth 1
        * ``transitively_affected`` — all predecessors beyond depth 1
        * ``affected_api_routes``   — affected nodes with API-route decorators
        * ``affected_files``        — unique file paths hosting affected nodes
        * ``impact_depth``          — maximum BFS depth reached
        """
        directly_affected: list[str] = []
        transitively_affected: list[str] = []
        affected_api_routes: list[str] = []
        affected_files_set: set[str] = set()
        max_depth_reached: int = 0

        if node_id not in self.graph:
            target_symbol = (
                self.graph.nodes[node_id].get("qualified_name", node_id)
                if node_id in self.graph
                else node_id
            )
            return ImpactAnalysisResult(
                target_symbol=target_symbol,
                directly_affected=[],
                transitively_affected=[],
                affected_api_routes=[],
                affected_files=[],
                impact_depth=0,
            )

        target_symbol: str = self.graph.nodes[node_id].get("qualified_name", node_id)

        visited: set[str] = {node_id}
        frontier: deque[tuple[str, int]] = deque([(node_id, 0)])

        while frontier:
            current, d = frontier.popleft()
            for predecessor in self.graph.predecessors(current):
                edge_data = self.graph.edges[predecessor, current]
                etype = edge_data.get("edge_type")
                if etype not in (_CALLS_EDGE, _IMPORTS_EDGE, _INHERITS_EDGE):
                    continue
                if predecessor in visited:
                    continue
                visited.add(predecessor)
                depth_reached = d + 1
                max_depth_reached = max(max_depth_reached, depth_reached)

                if depth_reached == 1:
                    directly_affected.append(predecessor)
                else:
                    transitively_affected.append(predecessor)

                # Collect file path
                pred_data = self.graph.nodes[predecessor]
                fp = pred_data.get("file_path")
                if fp:
                    affected_files_set.add(fp)

                # Check for API route
                if _is_api_route_node(pred_data):
                    affected_api_routes.append(predecessor)

                frontier.append((predecessor, depth_reached))

        return ImpactAnalysisResult(
            target_symbol=target_symbol,
            directly_affected=directly_affected,
            transitively_affected=transitively_affected,
            affected_api_routes=affected_api_routes,
            affected_files=sorted(affected_files_set),
            impact_depth=max_depth_reached,
        )

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------

    def find_entry_points(self, node_ids: list[str]) -> list[str]:
        """
        From *node_ids*, return those that have no incoming CALLS edges.

        These are "roots" — functions never called by any other node in the repo.
        """
        result: list[str] = []
        for nid in node_ids:
            if nid not in self.graph:
                continue
            has_caller = any(
                self.graph.edges[pred, nid].get("edge_type") == _CALLS_EDGE
                for pred in self.graph.predecessors(nid)
            )
            if not has_caller:
                result.append(nid)
        return result

    def find_api_route_nodes(self) -> list[str]:
        """Return all function nodes decorated as API routes."""
        return [
            nid
            for nid, data in self.graph.nodes(data=True)
            if data.get("node_type") == "function" and _is_api_route_node(data)
        ]

    def find_symbol_nodes(self, name_fragment: str) -> list[str]:
        """
        Case-insensitive substring match on ``name`` or ``qualified_name``.

        Returns a list of matching node_ids.
        """
        fragment_lower = name_fragment.lower()
        return [
            nid
            for nid, data in self.graph.nodes(data=True)
            if fragment_lower in data.get("name", "").lower()
            or fragment_lower in data.get("qualified_name", "").lower()
        ]

    # ------------------------------------------------------------------
    # Neighbour / edge queries
    # ------------------------------------------------------------------

    def get_neighbors_by_edge_type(
        self,
        node_id: str,
        edge_type: str,
        reverse: bool = False,
    ) -> list[str]:
        """
        Return direct neighbours connected by *edge_type*.

        Parameters
        ----------
        node_id:    source node
        edge_type:  one of DEFINES / IMPORTS / CALLS / INHERITS
        reverse:    if True, walk incoming edges (predecessors) instead
        """
        if node_id not in self.graph:
            return []

        if reverse:
            neighbors = self.graph.predecessors(node_id)
            edge_iter = ((pred, node_id) for pred in neighbors)
        else:
            neighbors = self.graph.successors(node_id)
            edge_iter = ((node_id, succ) for succ in neighbors)

        result: list[str] = []
        for src, dst in edge_iter:
            if not self.graph.has_edge(src, dst):
                continue
            if self.graph.edges[src, dst].get("edge_type") == edge_type:
                result.append(dst if not reverse else src)
        return result

    def shortest_path(self, src_node_id: str, dst_node_id: str) -> list[str]:
        """
        Return the shortest directed path from *src_node_id* to *dst_node_id*.

        Returns an empty list when no path exists or either node is absent.
        """
        if src_node_id not in self.graph or dst_node_id not in self.graph:
            return []
        try:
            return nx.shortest_path(self.graph, src_node_id, dst_node_id)
        except nx.NetworkXNoPath:
            return []
        except nx.NodeNotFound:
            return []

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def analyze_architecture(self) -> dict[str, Any]:
        """
        Analyze the repository's graph structure to identify architectural attributes:
        1. Circular imports (cycles in file-to-file IMPORTS edges)
        2. Tightly coupled modules (efferent/afferent coupling & Instability scores)
        3. Bedrock dependency hubs (high in-degree file/class/func nodes)
        4. Oversized orchestrators (high out-degree function nodes)
        5. Dead code candidates (non-entrypoint functions/classes with in-degree = 0)
        """
        import networkx as nx

        # Filter graphs
        file_graph = nx.DiGraph()
        call_graph = nx.DiGraph()

        for u, v, d in self.graph.edges(data=True):
            etype = d.get("edge_type")
            if etype == _IMPORTS_EDGE:
                file_graph.add_edge(u, v)
            elif etype == _CALLS_EDGE:
                call_graph.add_edge(u, v)

        # 1. Circular Imports
        circular_imports: list[list[str]] = []
        if file_graph.number_of_nodes() > 0:
            try:
                simple_cycles = list(nx.simple_cycles(file_graph))
                for cycle in simple_cycles[:10]:
                    circular_imports.append([self.graph.nodes[n].get("file_path", n) for n in cycle])
            except Exception:
                pass

        # 2. Instability & Coupling Scores per File
        file_metrics = {}
        for nid, data in self.graph.nodes(data=True):
            if data.get("node_type") == "file":
                ca = file_graph.in_degree(nid) if nid in file_graph else 0
                ce = file_graph.out_degree(nid) if nid in file_graph else 0
                
                instability = 0.0
                if ca + ce > 0:
                    instability = ce / (ca + ce)
                
                file_metrics[data.get("file_path", nid)] = {
                    "efferent_coupling": ce,
                    "afferent_coupling": ca,
                    "instability": round(instability, 3)
                }

        # 3. Dependency Hubs (High in-degree in CALLS edges)
        hubs = []
        for nid, data in self.graph.nodes(data=True):
            if data.get("node_type") in ("function", "class") and nid in call_graph:
                in_deg = call_graph.in_degree(nid)
                if in_deg >= 2:
                    hubs.append({
                        "qualified_name": data.get("qualified_name", ""),
                        "symbol_type": data.get("symbol_type", ""),
                        "in_degree": in_deg
                    })
        hubs.sort(key=lambda x: x["in_degree"], reverse=True)
        hubs = hubs[:10]

        # 4. Oversized Orchestrators (High out-degree in CALLS edges)
        orchestrators = []
        for nid, data in self.graph.nodes(data=True):
            if data.get("node_type") == "function" and nid in call_graph:
                out_deg = call_graph.out_degree(nid)
                if out_deg >= 3:
                    orchestrators.append({
                        "qualified_name": data.get("qualified_name", ""),
                        "symbol_type": data.get("symbol_type", ""),
                        "out_degree": out_deg
                    })
        orchestrators.sort(key=lambda x: x["out_degree"], reverse=True)
        orchestrators = orchestrators[:10]

        # 5. Dead Code Candidates (in-degree == 0, not entrypoint, not route)
        dead_candidates = []
        for nid, data in self.graph.nodes(data=True):
            if data.get("node_type") in ("function", "class") and not _is_api_route_node(data):
                in_deg = call_graph.in_degree(nid) if nid in call_graph else 0
                name_lower = data.get("name", "").lower()
                is_main = name_lower in ("main", "run", "start")
                has_decorators = len(data.get("decorators", [])) > 0
                
                if in_deg == 0 and not is_main and not has_decorators:
                    if "test_" not in data.get("file_path", ""):
                        dead_candidates.append({
                            "qualified_name": data.get("qualified_name", ""),
                            "symbol_type": data.get("symbol_type", ""),
                            "file_path": data.get("file_path", ""),
                            "line_start": data.get("line_start", 0)
                        })

        return {
            "circular_imports": circular_imports,
            "coupling_instability": file_metrics,
            "dependency_hubs": hubs,
            "oversized_orchestrators": orchestrators,
            "dead_code_candidates": dead_candidates[:15]
        }

    def node_count(self) -> int:
        """Total number of nodes in the graph."""
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        """Total number of edges in the graph."""
        return self.graph.number_of_edges()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """
        Serialise the graph to a JSON file using networkx node-link format.

        All node and edge attributes are preserved verbatim.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = nx.node_link_data(self.graph, edges="links")
        payload = {
            "graph": data,
            "_node_by_qualified": self._node_by_qualified,
            "_file_to_node": self._file_to_node,
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)

        logger.info(
            "CodeGraph saved → %s  (nodes=%d  edges=%d)",
            path,
            self.node_count(),
            self.edge_count(),
        )

    @classmethod
    def load(cls, path: Path) -> "CodeGraph":
        """
        Deserialise a CodeGraph from a JSON file written by :meth:`save`.

        Rebuilds ``_node_by_qualified`` and ``_file_to_node`` indexes from
        the stored payload (so they are available even if the stored indexes
        are stale or absent — they are re-derived from node attributes as
        a fallback).
        """
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)

        instance = cls()
        instance.graph = nx.node_link_graph(payload["graph"], directed=True, multigraph=False, edges="links")

        # Restore indexes from payload if present.
        instance._node_by_qualified = payload.get("_node_by_qualified", {})
        instance._file_to_node = payload.get("_file_to_node", {})

        # If indexes were missing, rebuild them from node attributes.
        if not instance._node_by_qualified or not instance._file_to_node:
            logger.warning("Index maps missing in saved graph — rebuilding from node attributes.")
            for nid, data in instance.graph.nodes(data=True):
                qn = data.get("qualified_name")
                if qn:
                    instance._node_by_qualified[qn] = nid
                fp = data.get("file_path")
                nt = data.get("node_type")
                if fp and nt == "file":
                    instance._file_to_node[_normalize_path(fp)] = nid

        logger.info(
            "CodeGraph loaded ← %s  (nodes=%d  edges=%d)",
            path,
            instance.node_count(),
            instance.edge_count(),
        )
        return instance
