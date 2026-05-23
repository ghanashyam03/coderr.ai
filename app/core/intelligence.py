from __future__ import annotations

"""
Core Intelligence — high-level orchestrator tying all components together.

Handles three operations:
    1. query(repo_name, question)
       Full pipeline: retrieve → assemble context → generate answer via Ollama

    2. inspect_function(repo_name, fn_name)
       Detailed structural report about a function:
       source, callers, callees, file, docstring, decorators

    3. get_dependencies(repo_name, symbol_name)
       Formatted dependency chain report using graph traversal

    4. analyze_impact(repo_name, symbol_name)
       Detailed impact analysis: what breaks if this symbol changes

All operations load the graph and registry from disk on first use per repo
and cache them in memory for the session lifetime.
"""

import logging
from pathlib import Path
from typing import Optional

from app.config.settings import settings
from app.embeddings.embedder import Embedder
from app.graph.graph_engine import CodeGraph
from app.ingestion.ingestion_pipeline import load_repo_artifacts
from app.llm.ollama_client import OllamaClient, OllamaConnectionError
from app.parsing.symbol_registry import SymbolRegistry
from app.reasoning.context_assembler import ContextAssembler
from app.retrieval.intent_router import classify_intent, extract_keywords
from app.retrieval.retrieval_engine import RetrievalEngine
from app.schemas.models import (
    ImpactAnalysisResult,
    QueryIntent,
    QueryResponse,
    SymbolType,
)
from app.vector_store.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_BASE = """You are a senior principal systems engineer and software architect.
You are walking through a Python codebase to explain its core structural, dependency, and execution systems.
You have been given a deeply detailed, hierarchical context containing project overview details, entry points, call tree diagrams, orchestrators, dependencies, and implementations.

Strict Rules of Engagement:
1. Ground every statement strictly in the provided code context. Do NOT generalize, invent, or hallucinate behaviors, files, or symbols.
2. Cite exact qualified names, file paths, and line numbers.
3. Adopt a highly technical, rigorous, and direct tone. Do NOT merely describe retrieved functions. Instead, explain the 'WHY' behind the architecture: explain orchestration responsibility, subsystem purpose, execution significance, architectural intent, downstream impact, and how components contribute to the system's runtime semantics.
4. Do NOT use generic chatbot conversational filler (e.g. "Sure! Here is...", "As an AI..."). Speak directly as a senior engineering lead.
5. If the context does not contain enough information, explain exactly what is missing and state the limits of the analyzed code.
"""

_SYSTEM_PROMPT_FLOW = _SYSTEM_PROMPT_BASE + """

When explaining the execution flow and call pathways:
- Identify the exact entry points (FastAPI route lifecycle, PyTorch forward execution, Typer/Click CLI commands, Celery task handlers).
- Map the execution chain structurally: entrypoint → orchestrator → services → utilities.
- Explain the causal mechanisms: WHAT controls the execution (e.g., event loops, framework lifecycles, main blocks), HOW data and state propagate across call boundaries, and WHERE orchestration decisions or routing logic reside.
- Visualize the call chain using a clean text-based flow diagram.
- Point out where complexity accumulates in the execution sequence and what potential bottlenecks or state mutations exist.
"""

_SYSTEM_PROMPT_IMPACT = _SYSTEM_PROMPT_BASE + """

When performing architectural impact analysis:
- Trace both the immediate and transitive downstream dependencies (incoming call edges and imports) across the dependency graph.
- Pinpoint exactly which API routes, services, CLI commands, or core orchestrators will be disrupted if this target symbol's contract or implementation changes.
- Explain the causal relationship: e.g. "Module A couples to Class B through attribute assignment, so changing B will break call C."
- Categorize downstream items by risk level and identify which components are the most dangerous or sensitive to modify.
- Synthesize a clear warning detailing the potential domino effects across the architectural layers.
"""

_SYSTEM_PROMPT_LOOKUP = _SYSTEM_PROMPT_BASE + """

When providing a symbol lookup and architectural walkthrough:
- State the exact file, starting line number, type, and docstring of the symbol.
- Explain its precise architectural purpose in the system (e.g., "This class encapsulates the DB connection lifecycle"). Do not just describe the code; explain WHY this component exists and what responsibility it serves.
- Walk through the implementation logic line-by-line, pointing out how it interacts with state, external libraries, framework features (like decorators or dependency injection), and downstream components.
- Highlight key stateful transitions, complexity hotspots, and risk factors in the implementation.
"""

_SYSTEM_PROMPT_DEPS = _SYSTEM_PROMPT_BASE + """

When explaining architectural dependencies:
- Detail the coupling index (afferent/efferent coupling) and instability scores of the containing modules.
- List direct call/import dependencies, explain the purpose of each relationship, and then outline the transitive dependencies.
- Group clearly by architectural role: External libraries/frameworks, Core orchestrators, Bedrock internal helpers, and data schemas.
- Identify where dependencies converge (bedrock hubs) and explain the system-wide significance of these hubs.
- Detect circular import pathways or tight coupling between subsystems, and explain why they are dangerous.
"""

_SYSTEM_PROMPT_GENERAL = _SYSTEM_PROMPT_BASE + """

Provide a causal architectural explanation of the system or query topic:
- Explain why these components exist, how they interact at runtime, and where orchestration occurs.
- Speak as a senior principal systems engineer, maintaining absolute technical rigor and focus.
"""


_INTENT_TO_SYSTEM: dict[QueryIntent, str] = {
    QueryIntent.FLOW_EXPLANATION: _SYSTEM_PROMPT_FLOW,
    QueryIntent.IMPACT_ANALYSIS: _SYSTEM_PROMPT_IMPACT,
    QueryIntent.SYMBOL_LOOKUP: _SYSTEM_PROMPT_LOOKUP,
    QueryIntent.DEPENDENCY_QUERY: _SYSTEM_PROMPT_DEPS,
    QueryIntent.GENERAL_QUERY: _SYSTEM_PROMPT_GENERAL,
}


class CodeIntelligence:
    """
    High-level intelligence layer — the main public API for Coderr.

    Manages per-repo caches of CodeGraph, SymbolRegistry, and RetrievalEngine.
    Components are loaded from disk on first access and cached for the session.

    Args:
        data_dir: Data directory (defaults to settings.CODERR_DATA_DIR).
        model: LLM model name override (defaults to settings.OLLAMA_MODEL).
        max_context_chars: Max context size for LLM prompts.
        max_retrieval_results: Max symbols retrieved per query.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        model: Optional[str] = None,
        max_context_chars: int | None = None,
        max_retrieval_results: int | None = None,
    ) -> None:
        self._data_dir = data_dir or settings.CODERR_DATA_DIR
        self._model = model or settings.OLLAMA_MODEL
        self._max_context_chars = max_context_chars or settings.MAX_CONTEXT_CHARS
        self._max_retrieval_results = max_retrieval_results or settings.MAX_RETRIEVAL_RESULTS

        # Shared components (model-level singletons)
        self._embedder: Embedder = Embedder()
        self._qdrant: QdrantStore = QdrantStore(
            data_dir=str(Path(self._data_dir) / "qdrant")
        )
        self._ollama: OllamaClient = OllamaClient(model=self._model)

        # Per-repo cache: repo_name → (graph, registry, retrieval_engine, assembler)
        self._repo_cache: dict[
            str,
            tuple[CodeGraph, SymbolRegistry, RetrievalEngine, ContextAssembler],
        ] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        repo_name: str,
        question: str,
        model: Optional[str] = None,
    ) -> QueryResponse:
        """
        Answer a natural language question about the codebase.

        Pipeline:
            1. Classify intent + extract target symbol
            2. Retrieve relevant symbols (4-stage pipeline)
            3. Assemble structured context (budgeted, sectioned)
            4. Build LLM prompt
            5. Generate answer via Ollama
            6. Return QueryResponse with answer + sources

        Args:
            repo_name: Repository name (must be already indexed).
            question: Natural language question.
            model: Override LLM model for this query.

        Returns:
            QueryResponse with answer, intent, sources, and metadata.

        Raises:
            FileNotFoundError: If repo is not indexed.
            OllamaConnectionError: If Ollama is unreachable.
        """
        graph, registry, retrieval_engine, assembler = self._load_repo(repo_name)

        # Classify intent
        intent, target_symbol = classify_intent(question)
        logger.info("Query intent: %s | target: %s", intent.value, target_symbol)

        # Retrieve
        results = retrieval_engine.retrieve(question, max_results=self._max_retrieval_results)

        if not results:
            logger.warning("No retrieval results for query: %s", question)
            return QueryResponse(
                answer=(
                    "I could not find relevant code for this query. "
                    "Ensure the repository is indexed and the query refers to code that exists."
                ),
                intent=intent,
                sources=[],
                context_chars=0,
                model_used=model or self._model,
            )

        # Assemble context
        context = assembler.assemble(results, question, intent, target_symbol)
        logger.info(
            "Context assembled — %d chars, %d sources, truncated=%s",
            context.total_chars,
            len(context.sources),
            context.truncated,
        )

        # Build prompt
        selected_model = model or self._model
        system_prompt = _INTENT_TO_SYSTEM.get(intent, _SYSTEM_PROMPT_GENERAL)
        user_prompt = self._build_user_prompt(question, context.full_text, repo_name)

        # Generate
        logger.info("Generating answer with model: %s", selected_model)
        answer = self._ollama.generate(
            prompt=user_prompt,
            system=system_prompt,
            model=selected_model,
        )

        return QueryResponse(
            answer=answer,
            intent=intent,
            sources=context.sources,
            context_chars=context.total_chars,
            model_used=selected_model,
        )

    def inspect_function(
        self,
        repo_name: str,
        fn_name: str,
    ) -> dict:
        """
        Return a detailed structural inspection of a function.

        Includes: source, file, line, docstring, decorators,
        args, return type, direct callers, direct callees.

        Args:
            repo_name: Repository name.
            fn_name: Function or method name (simple or qualified).

        Returns:
            Dict with symbol details, or error dict if not found.
        """
        graph, registry, _, _ = self._load_repo(repo_name)

        # Find symbol in registry
        candidates = registry.find_by_name(fn_name)
        if not candidates:
            # Try fragment match
            candidates = registry.find_by_name_fragment(fn_name)

        if not candidates:
            return {
                "error": f"Symbol '{fn_name}' not found in repository '{repo_name}'.",
                "suggestion": "Try 'python main.py list-repos' to see indexed repos.",
            }

        # Use the most specific match (prefer exact name)
        exact = [c for c in candidates if c.qualified_name.split(".")[-1] == fn_name]
        entry = exact[0] if exact else candidates[0]

        node_id = graph.get_node_id(entry.qualified_name)
        node_data = graph.get_node_data(node_id) if node_id else None

        # Get call graph info
        callers: list[str] = []
        callees: list[str] = []

        if node_id and node_data:
            callers = node_data.get("callers", [])
            callees_node_ids = graph.get_neighbors_by_edge_type(node_id, "CALLS")
            callees = [
                graph.get_node_data(nid).get("qualified_name", nid)
                for nid in callees_node_ids
                if graph.get_node_data(nid)
            ]

        return {
            "name": entry.qualified_name.split(".")[-1],
            "qualified_name": entry.qualified_name,
            "symbol_type": entry.symbol_type.value,
            "file_path": entry.file_path,
            "line_start": entry.line_start,
            "class_name": entry.class_name,
            "direct_callers": callers,
            "direct_callees": callees,
            "callers_count": len(callers),
            "callees_count": len(callees),
            "all_matches": [c.qualified_name for c in candidates],
        }

    def get_dependencies(
        self,
        repo_name: str,
        symbol_name: str,
        depth: int = 2,
        direction: str = "downstream",
    ) -> dict:
        """
        Return the dependency chain for a symbol.

        Args:
            repo_name: Repository name.
            symbol_name: Symbol to analyze (simple or qualified name).
            depth: BFS depth for dependency traversal.
            direction: 'downstream' (what this calls), 'upstream' (what calls this), 'bidirectional'.

        Returns:
            Dict with direct and transitive dependency lists and edge metadata.
        """
        graph, registry, _, _ = self._load_repo(repo_name)

        candidates = registry.find_by_name(symbol_name)
        if not candidates:
            candidates = registry.find_by_name_fragment(symbol_name)

        if not candidates:
            return {"error": f"Symbol '{symbol_name}' not found in '{repo_name}'."}

        entry = candidates[0]
        node_id = graph.get_node_id(entry.qualified_name)

        if not node_id:
            return {"error": f"Symbol '{entry.qualified_name}' not in graph."}

        # Resolve direct and transitive based on direction
        if direction == "upstream":
            direct_deps = graph.get_neighbors_by_edge_type(node_id, "CALLS", reverse=True)
            transitive_deps = graph.get_dependents(node_id, depth=depth)
        elif direction == "bidirectional":
            direct_down = graph.get_neighbors_by_edge_type(node_id, "CALLS", reverse=False)
            direct_up = graph.get_neighbors_by_edge_type(node_id, "CALLS", reverse=True)
            direct_deps = list(set(direct_down + direct_up))

            transitive_down = graph.get_dependencies(node_id, depth=depth)
            transitive_up = graph.get_dependents(node_id, depth=depth)
            transitive_deps = list(set(transitive_down + transitive_up))
        elif direction == "impact":
            impact_res = graph.get_impact(node_id)
            direct_deps = impact_res.directly_affected
            transitive_deps = impact_res.transitively_affected
        elif direction == "execution":
            from app.graph.flow_reconstructor import ExecutionFlowReconstructor
            reconstructor = ExecutionFlowReconstructor(graph)
            trace_data = reconstructor.trace_execution_flow(node_id, max_depth=depth + 1)
            steps = trace_data.get("steps", [])
            # depth 1 is direct, depth > 1 is transitive
            direct_deps = [step["node_id"] for step in steps if step["depth"] == 1]
            transitive_deps = [step["node_id"] for step in steps if step["depth"] > 1]
        else: # default downstream
            direct_deps = graph.get_neighbors_by_edge_type(node_id, "CALLS", reverse=False)
            transitive_deps = graph.get_dependencies(node_id, depth=depth)

        def node_ids_to_names(node_ids: list[str]) -> list[str]:
            names = []
            for nid in node_ids:
                data = graph.get_node_data(nid)
                if data:
                    names.append(data.get("qualified_name", nid))
            return names

        # Gather edge metadata for direct dependencies
        edge_meta = {}
        for nid in direct_deps:
            if graph.graph.has_edge(node_id, nid):
                edata = graph.graph.edges[node_id, nid]
            elif graph.graph.has_edge(nid, node_id):
                edata = graph.graph.edges[nid, node_id]
            else:
                edata = {}

            meta_qn = graph.get_node_data(nid).get("qualified_name", nid) if graph.get_node_data(nid) else nid
            edge_meta[meta_qn] = {
                "resolution_type": edata.get("resolution_type", "unresolved"),
                "confidence": edata.get("confidence", 0.0),
                "evidence": edata.get("evidence", ""),
                "provenance": edata.get("provenance", "none")
            }

        return {
            "symbol": entry.qualified_name,
            "file_path": entry.file_path,
            "direction": direction,
            "direct_dependencies": node_ids_to_names(direct_deps),
            "transitive_dependencies": node_ids_to_names(transitive_deps),
            "dependency_depth": depth,
            "edge_metadata": edge_meta,
        }


    def analyze_impact(
        self,
        repo_name: str,
        symbol_name: str,
    ) -> ImpactAnalysisResult:
        """
        Analyze what breaks if a symbol changes.

        Performs reverse BFS through the call graph to find all
        directly and transitively affected symbols, API routes, and files.

        Args:
            repo_name: Repository name.
            symbol_name: Symbol to analyze.

        Returns:
            ImpactAnalysisResult with affected symbols categorized by depth.
        """
        graph, registry, _, _ = self._load_repo(repo_name)

        candidates = registry.find_by_name(symbol_name)
        if not candidates:
            candidates = registry.find_by_name_fragment(symbol_name)

        if not candidates:
            return ImpactAnalysisResult(
                target_symbol=symbol_name,
                directly_affected=[],
                transitively_affected=[],
                affected_api_routes=[],
                affected_files=[],
                impact_depth=0,
            )

        entry = candidates[0]
        node_id = graph.get_node_id(entry.qualified_name)

        if not node_id:
            return ImpactAnalysisResult(
                target_symbol=entry.qualified_name,
                directly_affected=[],
                transitively_affected=[],
                affected_api_routes=[],
                affected_files=[],
                impact_depth=0,
            )

        logger.info("Analyzing impact of: %s", entry.qualified_name)
        return graph.get_impact(node_id)

    def analyze_architecture(self, repo_name: str) -> dict:
        """
        Analyze the codebase's graph structure to identify architectural attributes:
        circular dependencies, coupled modules, dependency hubs, oversized orchestrators,
        instability scores, and dead code candidates.
        """
        graph, _, _, _ = self._load_repo(repo_name)
        return graph.analyze_architecture()

    def format_impact_report(self, impact: ImpactAnalysisResult) -> str:
        """Format an ImpactAnalysisResult into a human-readable report."""
        lines = [
            f"\n🎯 Impact Analysis: {impact.target_symbol}",
            "=" * 60,
            f"\nImpact Depth: {impact.impact_depth} levels",
            f"\n📍 DIRECTLY AFFECTED ({len(impact.directly_affected)} symbols):",
        ]
        for sym in impact.directly_affected:
            lines.append(f"  → {sym}")

        lines.append(f"\n🌊 TRANSITIVELY AFFECTED ({len(impact.transitively_affected)} symbols):")
        for sym in impact.transitively_affected[:20]:  # cap at 20 for readability
            lines.append(f"  → {sym}")
        if len(impact.transitively_affected) > 20:
            lines.append(f"  ... and {len(impact.transitively_affected) - 20} more")

        lines.append(f"\n🚪 AFFECTED API ROUTES ({len(impact.affected_api_routes)}):")
        for route in impact.affected_api_routes:
            lines.append(f"  ⚡ {route}")

        lines.append(f"\n📁 AFFECTED FILES ({len(impact.affected_files)}):")
        for f in impact.affected_files:
            lines.append(f"  📄 {f}")

        if not (
            impact.directly_affected
            or impact.transitively_affected
            or impact.affected_api_routes
        ):
            lines.append("\n  ✅ No internal dependents found. This symbol appears to be a leaf.")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_repo(
        self,
        repo_name: str,
    ) -> tuple[CodeGraph, SymbolRegistry, RetrievalEngine, ContextAssembler]:
        """
        Load and cache repo artifacts (graph + registry + engines).

        Raises:
            FileNotFoundError: If the repo has not been indexed.
        """
        if repo_name in self._repo_cache:
            return self._repo_cache[repo_name]

        logger.info("Loading repository artifacts for '%s'...", repo_name)
        graph, registry = load_repo_artifacts(repo_name, self._data_dir)

        retrieval_engine = RetrievalEngine(
            graph=graph,
            embedder=self._embedder,
            qdrant_store=self._qdrant,
            repo_name=repo_name,
            max_results=self._max_retrieval_results,
        )

        assembler = ContextAssembler(
            graph=graph,
            max_chars=self._max_context_chars,
        )

        self._repo_cache[repo_name] = (graph, registry, retrieval_engine, assembler)
        logger.info("Repository '%s' loaded and cached.", repo_name)

        return self._repo_cache[repo_name]

    def _build_user_prompt(
        self,
        question: str,
        context: str,
        repo_name: str,
    ) -> str:
        """Build the full user prompt combining question and assembled context."""
        return (
            f"Repository: {repo_name}\n"
            f"Question: {question}\n\n"
            f"Relevant Code Context:\n"
            f"{'=' * 60}\n"
            f"{context}\n"
            f"{'=' * 60}\n\n"
            f"Based on the code context above, answer the question: {question}"
        )

    def clear_cache(self, repo_name: Optional[str] = None) -> None:
        """Clear the in-memory cache for a specific repo or all repos."""
        if repo_name:
            self._repo_cache.pop(repo_name, None)
        else:
            self._repo_cache.clear()
