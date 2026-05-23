from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING
import networkx as nx

if TYPE_CHECKING:
    from app.graph.graph_engine import CodeGraph

logger = logging.getLogger(__name__)


class ExecutionFlowReconstructor:
    """
    Reconstructs probable call pathways (entrypoint → orchestrator → service → utility)
    and computes ranking metrics (proximity, centrality, call frequency).
    """

    def __init__(self, graph: CodeGraph) -> None:
        self.graph = graph

    def is_likely_entrypoint(self, node_id: str) -> bool:
        """
        Check if a node ID fits any entrypoint heuristic.
        """
        if node_id not in self.graph.graph:
            return False
        data = self.graph.graph.nodes[node_id]
        if data.get("node_type") != "function":
            return False

        # Heuristic 1: Main block / entrypoint names
        name_lower = data.get("name", "").lower()
        if name_lower in ("main", "run", "start", "execute", "handler", "entrypoint"):
            return True

        # Heuristic 2: API routes (decorated)
        decorators = data.get("decorators", [])
        for dec in decorators:
            dec_lower = dec.lower()
            if any(kw in dec_lower for kw in ("route", "get", "post", "put", "delete", "patch", "head", "options", "websocket")):
                return True

        # Heuristic 3: Celery task or schedule decorators (e.g. @app.task, @shared_task)
        if any("task" in dec.lower() or "schedule" in dec.lower() for dec in decorators):
            return True

        # Heuristic 4: Graph Root (no incoming CALLS edges)
        has_caller = any(
            self.graph.graph.edges[pred, node_id].get("edge_type") == "CALLS"
            for pred in self.graph.graph.predecessors(node_id)
        )
        if not has_caller:
            return True

        return False

    def find_all_entrypoints(self) -> list[str]:
        """Find all nodes that qualify as likely entrypoints in the repository."""
        entrypoints = []
        for nid, data in self.graph.graph.nodes(data=True):
            if data.get("node_type") == "function" and self.is_likely_entrypoint(nid):
                entrypoints.append(nid)
        return entrypoints

    def calculate_entrypoint_proximity(self) -> dict[str, int]:
        """
        Run multi-source BFS starting from all entry points to compute
        the shortest path length (proximity) to every other function node.
        Returns a mapping of node_id -> proximity (depth). Unreachable is -1.
        """
        proximity = {}
        entrypoints = self.find_all_entrypoints()

        # Initialize
        for nid in self.graph.graph.nodes:
            proximity[nid] = -1

        queue = []
        for ep in entrypoints:
            proximity[ep] = 0
            queue.append((ep, 0))

        while queue:
            curr, dist = queue.pop(0)
            # Walk outgoing CALLS edges (forward call chain)
            for successor in self.graph.graph.successors(curr):
                if not self.graph.graph.has_edge(curr, successor):
                    continue
                etype = self.graph.graph.edges[curr, successor].get("edge_type")
                if etype == "CALLS":
                    if proximity[successor] == -1 or dist + 1 < proximity[successor]:
                        proximity[successor] = dist + 1
                        queue.append((successor, dist + 1))

        return proximity

    def trace_execution_flow(self, start_node_id: str, max_depth: int = 4) -> dict:
        """
        Reconstruct a probable forward execution flow starting from start_node_id.
        Walks forward CALLS edges using BFS and assigns execution roles:
            depth 0: Entrypoint
            depth 1: Orchestration
            depth 2: Services
            depth 3+: Utilities / Bedrock
        """
        if start_node_id not in self.graph.graph:
            return {"error": f"Start node {start_node_id} not found in graph."}

        flow_steps = []
        visited = set()
        # queue stores: (node_id, depth, caller_node_id, cumulative_confidence)
        queue = [(start_node_id, 0, None, 1.0)]

        while queue:
            curr, depth, caller, parent_conf = queue.pop(0)
            if depth > max_depth:
                continue
            if curr in visited:
                continue
            visited.add(curr)

            # Calculate confidence for this step
            confidence = parent_conf
            res_type = "direct"
            if caller is not None:
                if self.graph.graph.has_edge(caller, curr):
                    edge_data = self.graph.graph.edges[caller, curr]
                    res_type = edge_data.get("resolution_type", "unresolved")
                    if res_type in ("direct", "import"):
                        edge_conf = 1.0
                    elif res_type == "heuristic":
                        edge_conf = 0.7
                    else:
                        edge_conf = 0.3
                    confidence = parent_conf * edge_conf
                else:
                    confidence = parent_conf * 0.3
                    res_type = "unresolved"

            data = self.graph.graph.nodes[curr]
            role = "utility"
            if depth == 0:
                role = "entrypoint"
            elif depth == 1:
                role = "orchestrator"
            elif depth == 2:
                role = "service"

            step = {
                "node_id": curr,
                "name": data.get("name", ""),
                "qualified_name": data.get("qualified_name", ""),
                "file_path": data.get("file_path", ""),
                "line_start": data.get("line_start", 0),
                "depth": depth,
                "role": role,
                "caller": caller,
                "confidence": round(confidence, 2),
                "resolution_type": res_type,
            }
            flow_steps.append(step)

            # Sort successors by out-degree (high out-degree orchestrators first)
            successors = []
            for succ in self.graph.graph.successors(curr):
                if not self.graph.graph.has_edge(curr, succ):
                    continue
                etype = self.graph.graph.edges[curr, succ].get("edge_type")
                if etype == "CALLS":
                    out_deg = self.graph.graph.out_degree(succ)
                    successors.append((succ, out_deg))

            # Sort by out-degree descending
            successors.sort(key=lambda x: x[1], reverse=True)
            for succ, _ in successors:
                if succ not in visited:
                    queue.append((succ, depth + 1, curr, confidence))

        return {
            "start_node": start_node_id,
            "steps": flow_steps,
            "max_depth_reached": max(s["depth"] for s in flow_steps) if flow_steps else 0
        }
