from __future__ import annotations

"""
FastAPI REST API — lightweight HTTP interface for Coderr.

Endpoints:
    GET  /health
    GET  /repos
    POST /index
    POST /query
    GET  /dependencies/{repo_name}/{symbol}
    GET  /impact/{repo_name}/{symbol}
    GET  /inspect/{repo_name}/{fn_name}
"""

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config.settings import settings
from app.ingestion.ingestion_pipeline import IngestionPipeline, list_indexed_repos
from app.utils.logging_config import setup_logging

setup_logging(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Coderr — AI Codebase Intelligence",
    description=(
        "Local-first AI repository intelligence engine. "
        "Index Python repositories and query them with natural language."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class IndexRequest(BaseModel):
    repo_path: str = Field(..., description="Absolute or relative path to the Python repository.")
    repo_name: Optional[str] = Field(None, description="Optional repository name override.")


class IndexResponse(BaseModel):
    status: str
    repo_name: str
    file_count: int
    symbol_count: int
    duration_seconds: float


class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question about the codebase.")
    repo_name: str = Field(..., description="Repository name (must be already indexed).")
    model: Optional[str] = Field(None, description="Ollama model override.")


class QueryResponse(BaseModel):
    answer: str
    intent: str
    sources: list[str]
    context_chars: int
    model_used: str


class DepsResponse(BaseModel):
    symbol: str
    file_path: str
    direct_dependencies: list[str]
    transitive_dependencies: list[str]
    direction: Optional[str] = "downstream"
    dependency_depth: Optional[int] = 2
    edge_metadata: Optional[dict[str, dict]] = None


class ImpactResponse(BaseModel):
    target_symbol: str
    directly_affected: list[str]
    transitively_affected: list[str]
    affected_api_routes: list[str]
    affected_files: list[str]
    impact_depth: int


class InspectResponse(BaseModel):
    name: str
    qualified_name: str
    symbol_type: str
    file_path: str
    line_start: int
    class_name: Optional[str]
    direct_callers: list[str]
    direct_callees: list[str]


# ---------------------------------------------------------------------------
# Singleton intelligence instance (loaded once on startup)
# ---------------------------------------------------------------------------

_intelligence = None


def _get_intelligence():
    global _intelligence
    if _intelligence is None:
        from app.core.intelligence import CodeIntelligence
        _intelligence = CodeIntelligence()
    return _intelligence


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", tags=["System"])
def health() -> dict:
    """Health check — also verifies Ollama availability."""
    intel = _get_intelligence()
    ollama_ok = intel._ollama.check_available()
    return {
        "status": "ok",
        "ollama_available": ollama_ok,
        "ollama_model": settings.OLLAMA_MODEL,
        "data_dir": settings.CODERR_DATA_DIR,
    }


@app.get("/repos", tags=["Repositories"])
def list_repos() -> list[dict]:
    """List all indexed repositories."""
    return list_indexed_repos()


@app.post("/index", response_model=IndexResponse, tags=["Repositories"])
def index_repo(request: IndexRequest) -> IndexResponse:
    """
    Index a Python repository.

    This operation is synchronous and may take several minutes for large repos.
    """
    pipeline = IngestionPipeline()
    try:
        result = pipeline.index(
            repo_path=request.repo_path,
            repo_name=request.repo_name,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Indexing failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}")

    return IndexResponse(
        status="indexed",
        repo_name=result.get("repo_name", ""),
        file_count=result.get("file_count", 0),
        symbol_count=result.get("symbol_count", 0),
        duration_seconds=result.get("duration_seconds", 0.0),
    )


@app.post("/query", response_model=QueryResponse, tags=["Intelligence"])
def query_repo(request: QueryRequest) -> QueryResponse:
    """
    Query an indexed repository with a natural language question.

    Returns a grounded answer based on retrieved code context.
    """
    intel = _get_intelligence()
    try:
        response = intel.query(
            repo_name=request.repo_name,
            question=request.question,
            model=request.model,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("Query failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")

    return QueryResponse(
        answer=response.answer,
        intent=response.intent.value,
        sources=response.sources,
        context_chars=response.context_chars,
        model_used=response.model_used,
    )


@app.get("/dependencies/{repo_name}/{symbol}", response_model=DepsResponse, tags=["Intelligence"])
def get_dependencies(repo_name: str, symbol: str, depth: int = 2, direction: str = "downstream") -> DepsResponse:
    """Get the dependency chain for a symbol in a repository."""
    intel = _get_intelligence()
    try:
        result = intel.get_dependencies(
            repo_name=repo_name,
            symbol_name=symbol,
            depth=depth,
            direction=direction,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return DepsResponse(**result)


@app.get("/impact/{repo_name}/{symbol}", response_model=ImpactResponse, tags=["Intelligence"])
def get_impact(repo_name: str, symbol: str) -> ImpactResponse:
    """Analyze what breaks if a symbol changes."""
    intel = _get_intelligence()
    try:
        result = intel.analyze_impact(repo_name=repo_name, symbol_name=symbol)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return ImpactResponse(**result.model_dump())


@app.get("/inspect/{repo_name}/{fn_name}", response_model=InspectResponse, tags=["Intelligence"])
def inspect_function(repo_name: str, fn_name: str) -> InspectResponse:
    """Inspect a function: location, callers, callees, and signature."""
    intel = _get_intelligence()
    try:
        result = intel.inspect_function(repo_name=repo_name, fn_name=fn_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return InspectResponse(
        name=result.get("name", ""),
        qualified_name=result.get("qualified_name", ""),
        symbol_type=result.get("symbol_type", ""),
        file_path=result.get("file_path", ""),
        line_start=result.get("line_start", 0),
        class_name=result.get("class_name"),
        direct_callers=result.get("direct_callers", []),
        direct_callees=result.get("direct_callees", []),
    )


@app.get("/architecture/{repo_name}", tags=["Intelligence"])
def get_architecture(repo_name: str) -> dict:
    """
    Analyze the repository's graph structure to identify architectural attributes:
    circular dependencies, coupling metrics, hubs, oversized orchestrators, and dead code.
    """
    intel = _get_intelligence()
    try:
        return intel.analyze_architecture(repo_name=repo_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
