from __future__ import annotations

"""
Hybrid Retrieval Engine — 4-stage staged retrieval pipeline.

Stage 1 — Semantic Retrieval:
    Embed the query with bge-small. Search Qdrant for top-20 semantically
    similar symbols.

Stage 2 — Keyword / Exact Symbol Boost:
    Extract keyword tokens from the query. Search the graph for nodes whose
    name matches any token (exact or substring). Add these with source="keyword".

Stage 3 — Graph Dependency Expansion:
    For each result from Stage 1+2, traverse the graph based on query intent:
    - FLOW_EXPLANATION  → expand callees (depth=2) + callers (depth=1)
    - IMPACT_ANALYSIS   → expand dependents/callers (depth=3, reverse direction)
    - DEPENDENCY_QUERY  → expand dependencies (depth=3)
    - SYMBOL_LOOKUP     → no expansion (we already have the target)
    - GENERAL_QUERY     → expand calls (depth=1) + dependents (depth=1)

Stage 4 — Reranking:
    Deduplicate by symbol_id. Score each result:

    score = (
        semantic_score * 1.0
        + keyword_match_bonus * 0.4
        + (1.0 / (graph_depth + 1)) * 0.3
        + intent_alignment_score * 0.3
    )

    Intent alignment score (0.0–1.0):
    - FLOW_EXPLANATION  → +0.2 for async functions, +0.1 for decorated functions
    - SYMBOL_LOOKUP     → +0.5 if name exactly matches extracted symbol
    - IMPACT_ANALYSIS   → +0.2 for functions that are callers, +0.1 for API routes
    - DEPENDENCY_QUERY  → +0.2 for imported symbols, +0.1 for classes

Return top-N RetrievalResult sorted by final score.
"""

import logging
from typing import Optional, TYPE_CHECKING

from app.schemas.models import (
    CodeSymbol,
    QueryIntent,
    RetrievalResult,
    SymbolType,
)
from app.retrieval.intent_router import classify_intent, extract_keywords

if TYPE_CHECKING:
    from app.graph.graph_engine import CodeGraph
    from app.embeddings.embedder import Embedder
    from app.vector_store.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """
    Orchestrates the 4-stage hybrid retrieval pipeline.

    Args:
        graph: The CodeGraph for the indexed repository.
        embedder: Embedder for query vectorization.
        qdrant_store: QdrantStore for semantic search.
        repo_name: Name of the indexed repository.
        max_results: Maximum number of RetrievalResults to return.
    """

    def __init__(
        self,
        graph: "CodeGraph",
        embedder: "Embedder",
        qdrant_store: "QdrantStore",
        repo_name: str,
        max_results: int = 20,
    ) -> None:
        self.graph = graph
        self.embedder = embedder
        self.qdrant_store = qdrant_store
        self.repo_name = repo_name
        self.max_results = max_results

    def retrieve(
        self,
        query: str,
        max_results: Optional[int] = None,
    ) -> list[RetrievalResult]:
        """
        Execute the full 4-stage retrieval pipeline.

        Args:
            query: Natural language query string.
            max_results: Override max results count.

        Returns:
            Ranked list of RetrievalResult objects.
        """
        limit = max_results or self.max_results
        intent, target_symbol = classify_intent(query)

        logger.info(
            "Retrieval start — query='%s' intent=%s symbol=%s",
            query[:80],
            intent.value,
            target_symbol,
        )

        # Stage 1: Semantic retrieval
        stage1_results = self._stage1_semantic(query, top_k=20)
        logger.debug("Stage 1 (semantic): %d results", len(stage1_results))

        # Stage 2: Keyword / exact symbol boost
        keywords = extract_keywords(query)
        if target_symbol and target_symbol not in keywords:
            keywords.insert(0, target_symbol)
        stage2_results = self._stage2_keyword(keywords)
        logger.debug("Stage 2 (keyword): %d results", len(stage2_results))

        # Merge stage 1 + 2
        merged = self._merge_results(stage1_results, stage2_results)

        # Stage 3: Graph expansion
        stage3_results = self._stage3_graph_expand(merged, intent)
        logger.debug("Stage 3 (graph expand): %d total unique results", len(stage3_results))

        # Stage 4: Reranking
        final = self._stage4_rerank(stage3_results, intent, target_symbol)
        final = final[:limit]

        logger.info(
            "Retrieval complete — %d results returned (intent=%s)",
            len(final),
            intent.value,
        )
        return final

    # ------------------------------------------------------------------
    # Stage 1: Semantic retrieval
    # ------------------------------------------------------------------

    def _stage1_semantic(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        """Embed query and search Qdrant for most similar symbols."""
        try:
            query_vector = self.embedder.embed_query(query)
            raw_results = self.qdrant_store.search(self.repo_name, query_vector, top_k=top_k)
        except Exception as exc:
            logger.error("Semantic retrieval failed: %s", exc)
            return []

        results: list[RetrievalResult] = []
        for hit in raw_results:
            symbol = self._payload_to_symbol(hit.get("payload", {}))
            if symbol is None:
                continue
            results.append(
                RetrievalResult(
                    symbol=symbol,
                    score=float(hit.get("score", 0.0)),
                    source="semantic",
                    graph_depth=0,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Stage 2: Keyword / exact symbol matching
    # ------------------------------------------------------------------

    def _stage2_keyword(self, keywords: list[str]) -> list[RetrievalResult]:
        """Search the graph for nodes matching keyword tokens."""
        if not keywords:
            return []

        seen_node_ids: set[str] = set()
        results: list[RetrievalResult] = []

        for keyword in keywords:
            if len(keyword) < 2:
                continue

            matched_node_ids = self.graph.find_symbol_nodes(keyword)

            for node_id in matched_node_ids:
                if node_id in seen_node_ids:
                    continue
                seen_node_ids.add(node_id)

                node_data = self.graph.get_node_data(node_id)
                if node_data is None:
                    continue

                symbol = self._node_to_symbol(node_id, node_data)
                if symbol is None:
                    continue

                # Exact name match gets higher keyword score
                node_name = node_data.get("name", "").lower()
                keyword_score = 1.0 if node_name == keyword.lower() else 0.6

                results.append(
                    RetrievalResult(
                        symbol=symbol,
                        score=keyword_score,
                        source="keyword",
                        graph_depth=0,
                    )
                )

        return results

    # ------------------------------------------------------------------
    # Stage 3: Graph dependency expansion
    # ------------------------------------------------------------------

    def _is_low_value_leaf(self, node_id: str) -> bool:
        """Verify if a graph node is a low-value helper leaf node."""
        if not self.graph.graph.has_node(node_id):
            return True
        data = self.graph.graph.nodes[node_id]
        if data.get("node_type") != "function":
            return False
            
        # Leaf criteria: out-degree == 0, in-degree <= 1, no decorators
        try:
            out_deg = self.graph.graph.out_degree(node_id)
            in_deg = self.graph.graph.in_degree(node_id)
        except Exception:
            out_deg = 0
            in_deg = 0
            
        decorators = data.get("decorators", [])
        if out_deg == 0 and in_deg <= 1 and not decorators:
            return True
        return False

    def _stage3_graph_expand(
        self,
        base_results: list[RetrievalResult],
        intent: QueryIntent,
    ) -> list[RetrievalResult]:
        """Expand results by traversing graph neighbors based on intent."""
        if not base_results:
            return base_results

        all_results: dict[str, RetrievalResult] = {
            r.symbol.symbol_id: r for r in base_results
        }

        for result in base_results:
            node_id = self._symbol_id_to_node_id(result.symbol)
            if node_id is None:
                continue

            expanded_ids: list[tuple[str, int]] = []  # (node_id, depth)

            if intent == QueryIntent.FLOW_EXPLANATION:
                # Forward call chain (depth=2) + callers (depth=1)
                callees = self.graph.get_call_chain(node_id, depth=2)
                expanded_ids.extend((nid, i + 1) for i, nid in enumerate(callees[:10]))
                callers = self.graph.get_dependents(node_id, depth=1)
                expanded_ids.extend((nid, 1) for nid in callers[:5])

            elif intent == QueryIntent.IMPACT_ANALYSIS:
                # Reverse BFS — who depends on this? (depth=3)
                dependents = self.graph.get_dependents(node_id, depth=3)
                expanded_ids.extend((nid, i + 1) for i, nid in enumerate(dependents[:15]))

            elif intent == QueryIntent.DEPENDENCY_QUERY:
                # Forward deps (depth=3)
                deps = self.graph.get_dependencies(node_id, depth=3)
                expanded_ids.extend((nid, i + 1) for i, nid in enumerate(deps[:15]))

            elif intent == QueryIntent.SYMBOL_LOOKUP:
                # Minimal expansion: just immediate callers and callees
                callees = self.graph.get_call_chain(node_id, depth=1)
                expanded_ids.extend((nid, 1) for nid in callees[:3])

            else:  # GENERAL_QUERY
                # Balanced: calls (depth=1) + dependents (depth=1)
                callees = self.graph.get_call_chain(node_id, depth=1)
                expanded_ids.extend((nid, 1) for nid in callees[:5])
                dependents = self.graph.get_dependents(node_id, depth=1)
                expanded_ids.extend((nid, 1) for nid in dependents[:5])

            # Add expanded nodes to results
            for exp_node_id, depth in expanded_ids:
                # Aggressively suppress trivial helpers from graph expansion
                if self._is_low_value_leaf(exp_node_id):
                    continue

                exp_node_data = self.graph.get_node_data(exp_node_id)
                if exp_node_data is None:
                    continue

                exp_symbol = self._node_to_symbol(exp_node_id, exp_node_data)
                if exp_symbol is None:
                    continue

                if exp_symbol.symbol_id in all_results:
                    # Already present — update graph_depth if shallower
                    existing = all_results[exp_symbol.symbol_id]
                    if depth < existing.graph_depth or existing.graph_depth == 0:
                        all_results[exp_symbol.symbol_id] = RetrievalResult(
                            symbol=existing.symbol,
                            score=existing.score,
                            source=existing.source,
                            graph_depth=depth,
                        )
                else:
                    # Decay score by depth
                    base_score = 0.4 / (depth + 1)
                    all_results[exp_symbol.symbol_id] = RetrievalResult(
                        symbol=exp_symbol,
                        score=base_score,
                        source="graph",
                        graph_depth=depth,
                    )

        return list(all_results.values())

    # ------------------------------------------------------------------
    # Stage 4: Reranking
    # ------------------------------------------------------------------

    def _stage4_rerank(
        self,
        results: list[RetrievalResult],
        intent: QueryIntent,
        target_symbol: Optional[str],
    ) -> list[RetrievalResult]:
        """Apply final composite scoring and sort."""
        from app.graph.flow_reconstructor import ExecutionFlowReconstructor
        reconstructor = ExecutionFlowReconstructor(self.graph)
        proximity_map = reconstructor.calculate_entrypoint_proximity()

        # Build global confidence map from all entrypoints
        confidence_map: dict[str, float] = {}
        for ep in reconstructor.find_all_entrypoints():
            trace = reconstructor.trace_execution_flow(ep, max_depth=3)
            for step in trace.get("steps", []):
                nid = step["node_id"]
                conf = step["confidence"]
                if nid not in confidence_map or conf > confidence_map[nid]:
                    confidence_map[nid] = conf

        scored: list[tuple[float, RetrievalResult]] = []

        for result in results:
            sym = result.symbol

            # 1. Base Score from Retrieval Source
            if result.source == "semantic":
                source_base = result.score * 1.0
            elif result.source == "keyword":
                source_base = result.score * 0.8
            else:  # graph
                source_base = result.score * 0.6

            # 2. Multi-Signal Consensus Boost
            consensus_boost = 0.0
            if target_symbol:
                name_lower = sym.name.lower()
                target_lower = target_symbol.lower()
                if name_lower == target_lower:
                    consensus_boost += 0.5
                elif target_lower in name_lower or name_lower in target_lower:
                    consensus_boost += 0.2

            # 3. Graph Expansion Path Proximity
            graph_proximity = 0.0
            if result.graph_depth > 0:
                graph_proximity = (1.0 / (result.graph_depth + 1)) * 0.3

            # 4. Entrypoint Proximity Boost (Path distance from nearest entrypoint)
            entrypoint_proximity_boost = 0.0
            node_id = self._symbol_id_to_node_id(sym)
            if node_id and node_id in proximity_map:
                dist = proximity_map[node_id]
                if dist >= 0:
                    entrypoint_proximity_boost = (1.0 / (dist + 1)) * 0.4

            # 5. Direct Entry-Point Heuristics
            is_entry = False
            if any(
                d for d in sym.decorators
                if any(kw in d.lower() for kw in ("route", "get", "post", "put", "delete", "patch"))
            ):
                is_entry = True
            elif not sym.callers and sym.symbol_type in (SymbolType.FUNCTION, SymbolType.METHOD):
                is_entry = True

            entry_boost = 0.4 if is_entry else 0.0

            # 6. Orchestration Likelihood (Efferent coupling / out-degree)
            orchestration_likelihood_boost = 0.0
            if sym.calls and len(sym.calls) >= 2:
                orchestration_likelihood_boost += min(0.3, len(sym.calls) * 0.1)

            # 7. Dependency Importance (Afferent coupling / in-degree)
            dependency_importance_boost = 0.0
            if sym.callers and len(sym.callers) >= 2:
                dependency_importance_boost += min(0.2, len(sym.callers) * 0.05)

            # 7.5 Path Confidence Scoring (Heuristic paths get decayed)
            path_confidence_decay = 1.0
            if node_id and node_id in confidence_map:
                path_confidence_decay = confidence_map[node_id]

            # 8. Query-Intent Alignment Score
            intent_score = self._intent_alignment_score(result, intent, target_symbol)

            graph_boosts = graph_proximity + entrypoint_proximity_boost
            final_score = (
                source_base
                + consensus_boost
                + graph_boosts * path_confidence_decay
                + entry_boost
                + orchestration_likelihood_boost
                + dependency_importance_boost
                + intent_score * 0.4
            )

            scored.append((final_score, result))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            RetrievalResult(
                symbol=r.symbol,
                score=s,
                source=r.source,
                graph_depth=r.graph_depth,
            )
            for s, r in scored
        ]

    def _intent_alignment_score(
        self,
        result: RetrievalResult,
        intent: QueryIntent,
        target_symbol: Optional[str],
    ) -> float:
        """
        Compute intent alignment score (0.0–1.0) for a result.
        Boosts symbols that are structurally relevant to the query intent.
        """
        score = 0.0
        sym = result.symbol

        if intent == QueryIntent.SYMBOL_LOOKUP:
            if target_symbol:
                name_lower = sym.name.lower()
                target_lower = target_symbol.lower()
                if name_lower == target_lower:
                    score += 1.0
                elif target_lower in name_lower or name_lower in target_lower:
                    score += 0.5

        elif intent == QueryIntent.FLOW_EXPLANATION:
            # Prefer functions over classes
            if sym.symbol_type in (SymbolType.FUNCTION, SymbolType.METHOD):
                score += 0.3
            
            # Entry points are highly critical for tracing execution flows!
            is_entry = False
            if any(
                d for d in sym.decorators
                if any(kw in d.lower() for kw in ("route", "get", "post", "put", "delete", "patch"))
            ):
                is_entry = True
            elif not sym.callers and sym.symbol_type in (SymbolType.FUNCTION, SymbolType.METHOD):
                is_entry = True

            if is_entry:
                score += 0.6  # Heavy prioritization of entry points in flow explanation queries

            # Async functions are often HTTP handlers / high-level entry points
            node_data = self.graph.get_node_data(self._symbol_id_to_node_id(sym))
            if node_data and node_data.get("is_async"):
                score += 0.2
            # Decorated functions (routes, middlewares) are entry points
            if sym.decorators:
                score += 0.1

        elif intent == QueryIntent.IMPACT_ANALYSIS:
            # Prefer callers (things that use the target)
            if result.source in ("graph",) and result.graph_depth >= 1:
                score += 0.3
            # API route functions are high-impact callers
            if any(
                d for d in sym.decorators
                if any(kw in d.lower() for kw in ("route", "get", "post", "put", "delete", "patch"))
            ):
                score += 0.4

        elif intent == QueryIntent.DEPENDENCY_QUERY:
            # Prefer things that are imported/called
            if result.graph_depth >= 1:
                score += 0.2
            if sym.symbol_type == SymbolType.CLASS:
                score += 0.1

        return min(score, 1.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _merge_results(
        self,
        *result_lists: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """Merge multiple result lists, keeping highest score for duplicates."""
        merged: dict[str, RetrievalResult] = {}
        for results in result_lists:
            for r in results:
                sid = r.symbol.symbol_id
                if sid not in merged or r.score > merged[sid].score:
                    merged[sid] = r
        return list(merged.values())

    def _payload_to_symbol(self, payload: dict) -> Optional[CodeSymbol]:
        """Reconstruct a CodeSymbol from a Qdrant payload dict."""
        try:
            # The payload was stored with model_dump() minus full source
            # Restore required fields with defaults where needed
            return CodeSymbol(
                symbol_id=payload["symbol_id"],
                symbol_type=SymbolType(payload["symbol_type"]),
                name=payload["name"],
                qualified_name=payload["qualified_name"],
                file_path=payload["file_path"],
                line_start=payload.get("line_start", 0),
                line_end=payload.get("line_end", 0),
                source=payload.get("source_preview", ""),
                docstring=payload.get("docstring"),
                class_name=payload.get("class_name"),
                repo_name=payload.get("repo_name", self.repo_name),
                calls=payload.get("calls", []),
                callers=payload.get("callers", []),
                bases=payload.get("bases", []),
                decorators=payload.get("decorators", []),
            )
        except (KeyError, ValueError) as exc:
            logger.debug("Cannot reconstruct symbol from payload: %s", exc)
            return None

    def _node_to_symbol(self, node_id: str, node_data: dict) -> Optional[CodeSymbol]:
        """Build a minimal CodeSymbol from graph node data."""
        try:
            node_type = node_data.get("node_type", "function")
            if node_type == "file":
                return None  # We don't embed files

            symbol_type_str = node_data.get("symbol_type", "function")
            try:
                symbol_type = SymbolType(symbol_type_str)
            except ValueError:
                symbol_type = SymbolType.FUNCTION

            qualified_name = node_data.get("qualified_name", "")
            name = node_data.get("name", qualified_name.split(".")[-1])
            file_path = node_data.get("file_path", "")
            class_name = node_data.get("class_name")

            # Build symbol_id consistently with how ingestion builds it
            if class_name:
                symbol_id = f"{file_path}::{class_name}::{name}"
            else:
                symbol_id = f"{file_path}::{name}"

            return CodeSymbol(
                symbol_id=symbol_id,
                symbol_type=symbol_type,
                name=name,
                qualified_name=qualified_name,
                file_path=file_path,
                line_start=node_data.get("line_start", 0),
                line_end=0,
                source="",  # source not stored in graph nodes
                docstring=None,
                class_name=class_name,
                repo_name=self.repo_name,
                calls=[],
                callers=[],
                bases=[],
                decorators=node_data.get("decorators", []),
            )
        except Exception as exc:
            logger.debug("Cannot build symbol from node %s: %s", node_id, exc)
            return None

    def _symbol_id_to_node_id(self, symbol: CodeSymbol) -> Optional[str]:
        """Map a CodeSymbol to its graph node_id via qualified_name lookup."""
        node_id = self.graph.get_node_id(symbol.qualified_name)
        return node_id
