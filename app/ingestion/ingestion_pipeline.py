from __future__ import annotations

"""
Ingestion Pipeline — orchestrates the full repository indexing workflow.

Pipeline:
    1. Scanner     → discover all .py files
    2. AST Parser  → parse each file (parallel, ThreadPoolExecutor)
    3. Symbol Registry → build cross-file resolution table
    4. Graph Engine → build dependency/call graph
    5. Persistence  → save graph + registry to disk
    6. Symbol Flat  → flatten functions + classes → CodeSymbol list
    7. Embedder     → batch embed all CodeSymbols
    8. Qdrant Store → upsert vectors + metadata

Index state is stored in:
    {CODERR_DATA_DIR}/{repo_name}/
        graph.json        ← serialized CodeGraph
        registry.json     ← serialized SymbolRegistry
        index.json        ← indexing metadata (file count, symbol count, timestamp)

Re-indexing:
    If the collection already exists in Qdrant, it is deleted and rebuilt from scratch.
    This ensures the index always reflects the current state of the repository.
"""

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.config.settings import settings
from app.embeddings.embedder import Embedder
from app.graph.graph_engine import CodeGraph
from app.parsing.ast_parser import parse_file
from app.parsing.scanner import scan_repository
from app.parsing.symbol_registry import SymbolRegistry, build_registry
from app.schemas.models import (
    CodeSymbol,
    ParsedClass,
    ParsedFile,
    ParsedFunction,
    SymbolType,
)
from app.vector_store.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)


def _make_symbol_id(file_path: str, name: str, class_name: Optional[str] = None) -> str:
    """Build a deterministic symbol_id from its components."""
    if class_name:
        return f"{file_path}::{class_name}::{name}"
    return f"{file_path}::{name}"


def _function_to_symbol(
    fn: ParsedFunction,
    repo_name: str,
    symbol_type: SymbolType,
) -> CodeSymbol:
    """Convert a ParsedFunction to a CodeSymbol."""
    return CodeSymbol(
        symbol_id=_make_symbol_id(fn.file_path, fn.name, fn.class_name),
        symbol_type=symbol_type,
        name=fn.name,
        qualified_name=fn.qualified_name,
        file_path=fn.file_path,
        line_start=fn.line_start,
        line_end=fn.line_end,
        source=fn.source,
        docstring=fn.docstring,
        class_name=fn.class_name,
        repo_name=repo_name,
        calls=[c.resolved for c in fn.calls if c.resolved],
        callers=[],  # populated after graph build
        bases=[],
        decorators=fn.decorators,
    )


def _class_to_symbol(cls: ParsedClass, repo_name: str) -> CodeSymbol:
    """Convert a ParsedClass to a CodeSymbol."""
    return CodeSymbol(
        symbol_id=_make_symbol_id(cls.file_path, cls.name),
        symbol_type=SymbolType.CLASS,
        name=cls.name,
        qualified_name=cls.qualified_name,
        file_path=cls.file_path,
        line_start=cls.line_start,
        line_end=cls.line_end,
        source=cls.source,
        docstring=cls.docstring,
        class_name=None,
        repo_name=repo_name,
        calls=[],
        callers=[],
        bases=cls.bases,
        decorators=cls.decorators,
    )


def flatten_symbols(parsed_files: list[ParsedFile], repo_name: str) -> list[CodeSymbol]:
    """
    Convert all parsed functions, methods, and classes into CodeSymbol objects.

    NOTE: No file-level symbols are created. Only functions, methods, and classes.
    """
    symbols: list[CodeSymbol] = []

    for pf in parsed_files:
        if pf.parse_error:
            continue

        # Top-level functions
        for fn in pf.functions:
            symbols.append(_function_to_symbol(fn, repo_name, SymbolType.FUNCTION))

        for cls in pf.classes:
            # Class itself
            symbols.append(_class_to_symbol(cls, repo_name))

            # Methods
            for method in cls.methods:
                symbols.append(_function_to_symbol(method, repo_name, SymbolType.METHOD))

    return symbols


def _populate_callers_from_graph(symbols: list[CodeSymbol], graph: CodeGraph) -> None:
    """
    Fill in the callers field on each CodeSymbol using the graph's reverse edges.

    The graph engine already tracks callers on nodes. We sync that back to
    our CodeSymbol objects so they are stored in Qdrant for retrieval.
    """
    qn_to_symbol: dict[str, CodeSymbol] = {s.qualified_name: s for s in symbols}

    for sym in symbols:
        node_id = graph.get_node_id(sym.qualified_name)
        if node_id is None:
            continue
        node_data = graph.get_node_data(node_id)
        if node_data is None:
            continue
        caller_qns: list[str] = node_data.get("callers", [])
        sym.callers = caller_qns


class IngestionPipeline:
    """
    Orchestrates the complete repository indexing pipeline.

    Args:
        data_dir: Path where all indexed data is stored.
        embedder: Embedder instance (shared to avoid reloading the model).
        qdrant_store: QdrantStore instance.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        embedder: Optional[Embedder] = None,
        qdrant_store: Optional[QdrantStore] = None,
    ) -> None:
        self._data_dir = Path(data_dir or settings.CODERR_DATA_DIR)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        qdrant_path = str(self._data_dir / "qdrant")
        self.qdrant_store = qdrant_store or QdrantStore(data_dir=qdrant_path)
        self.embedder = embedder or Embedder()

    def _compile_repo_profile(
        self,
        parsed_files: list[ParsedFile],
        graph: CodeGraph,
        repo_path: Path,
    ) -> dict:
        """Compile a repository structural profile with architectural roles/summaries."""
        from app.graph.flow_reconstructor import ExecutionFlowReconstructor
        
        # 1. Gather file, class, function, method counts and external imports
        file_count = len(parsed_files)
        class_count = 0
        function_count = 0
        method_count = 0
        external_imports = set()
        
        for pf in parsed_files:
            class_count += len(pf.classes)
            function_count += len(pf.functions)
            for cls in pf.classes:
                method_count += len(cls.methods)
                
            for imp in pf.imports:
                if imp.module:
                    parts = imp.module.split(".")
                    external_imports.add(parts[0])
        
        # Filter commonly known libraries/frameworks
        known_frameworks = {
            "fastapi": "FastAPI Web Framework",
            "flask": "Flask Web Framework",
            "django": "Django Framework",
            "click": "Click CLI",
            "typer": "Typer CLI",
            "argparse": "Argparse CLI",
            "pytest": "PyTest Testing",
            "unittest": "Unittest Testing",
            "networkx": "NetworkX Graph Library",
            "qdrant_client": "Qdrant Vector Database Client",
            "sentence_transformers": "Sentence Transformers",
            "pydantic": "Pydantic Data Validation",
            "ollama": "Ollama LLM",
        }
        
        detected_frameworks = []
        for imp in external_imports:
            if imp in known_frameworks:
                detected_frameworks.append(known_frameworks[imp])
                
        frameworks_str = ", ".join(detected_frameworks) if detected_frameworks else "Standard Python library modules"
        
        # 2. Get entry points and core orchestrators from reconstructor/graph
        reconstructor = ExecutionFlowReconstructor(graph)
        raw_eps = reconstructor.find_all_entrypoints()
        
        entry_points = []
        for ep in raw_eps:
            data = graph.get_node_data(ep)
            if data:
                entry_points.append({
                    "qualified_name": data.get("qualified_name", ""),
                    "file_path": data.get("file_path", ""),
                    "symbol_type": data.get("symbol_type", ""),
                    "decorators": data.get("decorators", [])
                })
        
        # Core orchestrators (high out-degree symbols)
        architectural_analysis = graph.analyze_architecture()
        core_orchestrators = architectural_analysis.get("oversized_orchestrators", [])
        
        # 3. Compile module roles (by directories)
        dirs = {}
        for pf in parsed_files:
            try:
                rel_path = Path(pf.path).relative_to(repo_path)
                parent = str(rel_path.parent).replace("\\", "/")
            except ValueError:
                parent = "root"
            if parent == ".":
                parent = "root"
            if parent not in dirs:
                dirs[parent] = []
            dirs[parent].append(pf)
            
        module_roles = {}
        for dirname, files in dirs.items():
            # Heuristic naming and size analysis
            role = "Supporting codebase modules."
            dir_lower = dirname.lower()
            if dir_lower == "root":
                role = "Repository root level files including startup or config scripts."
            elif "test" in dir_lower:
                role = "Test suite housing unit, integration, or system tests."
            elif "api" in dir_lower or "router" in dir_lower or "routes" in dir_lower:
                role = "API routing and endpoint handler layer."
            elif "parsing" in dir_lower or "parser" in dir_lower:
                role = "Source scanning, parsing, and AST compilation layer."
            elif "graph" in dir_lower:
                role = "Code symbol relation and graph-based traversal layer."
            elif "schemas" in dir_lower or "models" in dir_lower:
                role = "Pydantic and data models mapping layer."
            elif "core" in dir_lower or "logic" in dir_lower or "services" in dir_lower:
                role = "Core business logic, services, and orchestrators."
            elif "config" in dir_lower or "settings" in dir_lower:
                role = "System configuration, environment loading, and settings."
            elif "vector_store" in dir_lower or "vector" in dir_lower:
                role = "Vector database storage and indexing interface."
            elif "embedding" in dir_lower:
                role = "Semantic sentence embedding and text representation layer."
            elif "retrieval" in dir_lower:
                role = "Hybrid semantic and graph retrieval layer."
            elif "reasoning" in dir_lower:
                role = "Context builder, prompt assembly, and reasoning layer."
            elif "cli" in dir_lower:
                role = "Command-Line Interface (CLI) entrypoint definitions and UI command routing."
            module_roles[dirname] = f"{role} ({len(files)} files)"
            
        # 4. Generate high-level system overview
        system_overview = (
            f"This codebase is a Python application structured with {file_count} modules containing "
            f"{class_count} classes and {function_count + method_count} functions/methods (including {method_count} class methods). "
            f"Key technologies/frameworks utilized: {frameworks_str}. "
            f"It features {len(entry_points)} detected entry points and {len(core_orchestrators)} primary orchestrator modules."
        )
        
        return {
            "system_overview": system_overview,
            "module_roles": module_roles,
            "core_orchestrators": core_orchestrators,
            "entry_points": entry_points,
        }

    def index(
        self,
        repo_path: str | Path,
        repo_name: Optional[str] = None,
        max_workers: Optional[int] = None,
    ) -> dict:
        """
        Index a Python repository end-to-end.

        Args:
            repo_path: Path to the repository root.
            repo_name: Name used for the Qdrant collection and data storage.
                       Defaults to the repository directory name.
            max_workers: Number of parallel parser threads.
                         Defaults to min(8, cpu_count).

        Returns:
            Summary dict with file_count, symbol_count, error_count, duration_seconds.
        """
        start_time = time.monotonic()
        root = Path(repo_path).resolve()

        if repo_name is None:
            repo_name = root.name

        repo_data_dir = self._data_dir / repo_name
        repo_data_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("Indexing repository: %s", root)
        logger.info("Repo name: %s", repo_name)
        logger.info("=" * 60)

        # ------ Step 1: Scan ------
        logger.info("[1/7] Scanning for Python files...")
        file_paths = scan_repository(root)

        if not file_paths:
            logger.warning("No Python files found in %s", root)
            return {"file_count": 0, "symbol_count": 0, "error_count": 0, "duration_seconds": 0}

        logger.info("Found %d Python files", len(file_paths))

        # ------ Step 2: Parse (parallel) ------
        workers = max_workers or min(8, (os.cpu_count() or 4))
        logger.info("[2/7] Parsing files with %d workers...", workers)

        parsed_files: list[ParsedFile] = []
        parse_errors = 0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_path = {
                executor.submit(parse_file, fp, root): fp for fp in file_paths
            }
            for future in as_completed(future_to_path):
                fp = future_to_path[future]
                try:
                    pf = future.result()
                    parsed_files.append(pf)
                    if pf.parse_error:
                        parse_errors += 1
                        logger.warning("Parse error in %s: %s", fp.name, pf.parse_error)
                except Exception as exc:
                    parse_errors += 1
                    logger.error("Fatal parse error for %s: %s", fp, exc)

        successful = len(parsed_files) - parse_errors
        logger.info(
            "Parsed %d/%d files successfully (%d errors)",
            successful,
            len(parsed_files),
            parse_errors,
        )

        # ------ Step 3: Build Symbol Registry ------
        logger.info("[3/7] Building symbol registry...")
        registry = build_registry(parsed_files)
        logger.info(
            "Registry built — %d symbols indexed",
            len(registry.get_all_symbols()),
        )

        # ------ Step 4: Build Graph ------
        logger.info("[4/7] Building dependency graph...")
        graph = CodeGraph()
        graph.build(parsed_files, registry)
        logger.info(
            "Graph built — %d nodes, %d edges",
            graph.node_count(),
            graph.edge_count(),
        )

        # ------ Step 5: Persist graph + registry ------
        logger.info("[5/7] Saving graph, registry, and repository structural profile...")
        graph_path = repo_data_dir / "graph.json"
        registry_path = repo_data_dir / "registry.json"
        profile_path = repo_data_dir / "profile.json"

        graph.save(graph_path)
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry.to_dict(), f, indent=2)

        # Build and save structural profile
        profile_data = self._compile_repo_profile(parsed_files, graph, root)
        with open(profile_path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=2)

        logger.info("Graph saved to %s", graph_path)
        logger.info("Registry saved to %s", registry_path)
        logger.info("Repository profile saved to %s", profile_path)

        # ------ Step 6: Flatten symbols ------
        logger.info("[6/7] Flattening symbols for embedding...")
        symbols = flatten_symbols(parsed_files, repo_name)

        # Back-fill callers from graph
        _populate_callers_from_graph(symbols, graph)

        logger.info(
            "Flattened %d symbols (functions: %d, methods: %d, classes: %d)",
            len(symbols),
            sum(1 for s in symbols if s.symbol_type.value == "function"),
            sum(1 for s in symbols if s.symbol_type.value == "method"),
            sum(1 for s in symbols if s.symbol_type.value == "class"),
        )

        if not symbols:
            logger.warning("No symbols to embed. Index complete but empty.")
            return {
                "file_count": len(file_paths),
                "symbol_count": 0,
                "error_count": parse_errors,
                "duration_seconds": time.monotonic() - start_time,
            }

        # ------ Step 7: Embed + Store ------
        logger.info("[7/7] Embedding and storing in Qdrant...")

        # Delete existing collection to avoid stale data
        if self.qdrant_store.collection_exists(repo_name):
            logger.info("Dropping existing collection for repo '%s'...", repo_name)
            self.qdrant_store.delete_collection(repo_name)

        self.qdrant_store.ensure_collection(repo_name)

        # Embed in batches
        logger.info("Embedding %d symbols (model: %s)...", len(symbols), self.embedder._model_name)
        vectors = self.embedder.embed_symbols_batch(
            symbols, batch_size=settings.EMBEDDING_BATCH_SIZE
        )

        # Upsert to Qdrant
        self.qdrant_store.upsert_symbols(repo_name, symbols, vectors)

        # Save index metadata
        duration = time.monotonic() - start_time
        index_meta = {
            "repo_name": repo_name,
            "repo_path": str(root),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "file_count": len(file_paths),
            "parsed_count": successful,
            "error_count": parse_errors,
            "symbol_count": len(symbols),
            "node_count": graph.node_count(),
            "edge_count": graph.edge_count(),
            "duration_seconds": round(duration, 2),
        }
        index_path = repo_data_dir / "index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_meta, f, indent=2)

        logger.info("=" * 60)
        logger.info("Indexing complete!")
        logger.info("  Files:    %d (%d errors)", len(file_paths), parse_errors)
        logger.info("  Symbols:  %d", len(symbols))
        logger.info("  Graph:    %d nodes, %d edges", graph.node_count(), graph.edge_count())
        logger.info("  Duration: %.1fs", duration)
        logger.info("=" * 60)

        return index_meta


def list_indexed_repos(data_dir: Optional[str] = None) -> list[dict]:
    """
    List all repositories that have been indexed.

    Returns:
        List of index metadata dicts for each indexed repo.
    """
    base = Path(data_dir or settings.CODERR_DATA_DIR)
    repos: list[dict] = []

    if not base.exists():
        return repos

    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        index_file = entry / "index.json"
        if index_file.exists():
            try:
                with open(index_file, encoding="utf-8") as f:
                    meta = json.load(f)
                repos.append(meta)
            except Exception:
                pass

    return repos


def load_repo_artifacts(
    repo_name: str,
    data_dir: Optional[str] = None,
) -> tuple[CodeGraph, SymbolRegistry]:
    """
    Load the CodeGraph and SymbolRegistry for an indexed repository.

    Args:
        repo_name: Repository name (as used during indexing).
        data_dir: Data directory override.

    Returns:
        (CodeGraph, SymbolRegistry) tuple.

    Raises:
        FileNotFoundError: If the repository has not been indexed.
    """
    base = Path(data_dir or settings.CODERR_DATA_DIR)
    repo_dir = base / repo_name

    graph_path = repo_dir / "graph.json"
    registry_path = repo_dir / "registry.json"

    if not graph_path.exists():
        raise FileNotFoundError(
            f"Repository '{repo_name}' has not been indexed. "
            f"Run: python main.py index <path_to_repo>"
        )

    logger.info("Loading graph from %s", graph_path)
    graph = CodeGraph.load(graph_path)

    logger.info("Loading registry from %s", registry_path)
    with open(registry_path, encoding="utf-8") as f:
        registry_data = json.load(f)
    registry = SymbolRegistry.from_dict(registry_data)

    return graph, registry
