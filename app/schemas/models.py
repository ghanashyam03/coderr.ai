from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SymbolType(str, Enum):
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"


class QueryIntent(str, Enum):
    SYMBOL_LOOKUP = "symbol_lookup"        # "where is X defined"
    FLOW_EXPLANATION = "flow_explanation"  # "explain authentication flow"
    IMPACT_ANALYSIS = "impact_analysis"    # "what breaks if I modify X"
    DEPENDENCY_QUERY = "dependency_query"  # "what does X depend on"
    GENERAL_QUERY = "general_query"        # fallback


class ResolutionType(str, Enum):
    DIRECT = "direct"             # Local calls resolved in the same module
    IMPORT = "import"             # Cross-module calls resolved via explicit imports
    HEURISTIC = "heuristic"       # Instance attribute or local variable heuristic resolutions
    UNRESOLVED = "unresolved"     # Static call that could not be verified in the registry


# ---------------------------------------------------------------------------
# AST Parsing Models
# ---------------------------------------------------------------------------


class ParsedImport(BaseModel):
    """One import statement in a file."""

    module: str
    names: list[str] = Field(default_factory=list)
    alias: Optional[str] = None  # import X as Y  (Y is alias)
    line: int = 0
    is_from: bool = False  # True for "from X import Y"


class ParsedCall(BaseModel):
    """One function/method call detected inside a function body."""

    name: str             # raw name as written in source: "validate_token" or "jwt.decode"
    line: int = 0
    resolved: Optional[str] = None  # fully-qualified name after symbol registry resolution
    resolution_type: ResolutionType = ResolutionType.UNRESOLVED


class ParsedFunction(BaseModel):
    """An extracted function or method from AST."""

    name: str
    qualified_name: str   # module.ClassName.method or module.function
    file_path: str
    line_start: int
    line_end: int
    source: str           # full source text of the function
    docstring: Optional[str] = None
    args: list[str] = Field(default_factory=list)
    return_annotation: Optional[str] = None
    decorators: list[str] = Field(default_factory=list)
    calls: list[ParsedCall] = Field(default_factory=list)
    is_async: bool = False
    is_method: bool = False
    class_name: Optional[str] = None
    assignments: dict[str, str] = Field(default_factory=dict)



class ParsedClass(BaseModel):
    """An extracted class from AST."""

    name: str
    qualified_name: str   # module.ClassName
    file_path: str
    line_start: int
    line_end: int
    source: str
    docstring: Optional[str] = None
    bases: list[str] = Field(default_factory=list)  # parent class names (raw)
    methods: list[ParsedFunction] = Field(default_factory=list)
    decorators: list[str] = Field(default_factory=list)
    assignments: dict[str, str] = Field(default_factory=dict)



class ParsedFile(BaseModel):
    """Result of parsing a single .py file."""

    path: str
    module_name: str       # dot-separated, e.g. "auth.jwt"
    imports: list[ParsedImport] = Field(default_factory=list)
    functions: list[ParsedFunction] = Field(default_factory=list)  # top-level only
    classes: list[ParsedClass] = Field(default_factory=list)
    parse_error: Optional[str] = None


# ---------------------------------------------------------------------------
# Symbol Registry Models
# ---------------------------------------------------------------------------


class SymbolRegistryEntry(BaseModel):
    """Global registry entry for a single defined symbol."""

    qualified_name: str          # auth.jwt.validate_token
    symbol_type: SymbolType
    file_path: str
    class_name: Optional[str] = None
    line_start: int = 0


class ImportResolution(BaseModel):
    """Maps a local name used in a file to its fully-qualified symbol."""

    local_name: str              # name as used in the file
    qualified_name: str          # full qualified name
    source_module: str           # module it was imported from


# ---------------------------------------------------------------------------
# Embedding / Storage Models
# ---------------------------------------------------------------------------


class CodeSymbol(BaseModel):
    """
    Unified model for a single embeddable code unit.
    Represents a function, method, or class — never a whole file.
    """

    symbol_id: str               # unique key: "{file_path}::{class_name}::{name}" or "{file_path}::{name}"
    symbol_type: SymbolType
    name: str
    qualified_name: str
    file_path: str
    line_start: int
    line_end: int
    source: str                  # full source text
    docstring: Optional[str] = None
    class_name: Optional[str] = None
    repo_name: str

    # Graph relationships (populated after graph build)
    calls: list[str] = Field(default_factory=list)      # resolved qualified names this symbol calls
    callers: list[str] = Field(default_factory=list)    # resolved qualified names that call this symbol
    bases: list[str] = Field(default_factory=list)      # for classes: parent class names
    decorators: list[str] = Field(default_factory=list)
    assignments: dict[str, str] = Field(default_factory=dict)



# ---------------------------------------------------------------------------
# Retrieval Models
# ---------------------------------------------------------------------------


class RetrievalResult(BaseModel):
    """One retrieved symbol with its retrieval metadata."""

    symbol: CodeSymbol
    score: float
    source: str   # "semantic" | "keyword" | "graph" | "dependency"
    graph_depth: int = 0


class AssembledContext(BaseModel):
    """The structured context assembled for an LLM prompt."""

    sections: dict[str, str] = Field(default_factory=dict)  # section_name -> formatted block
    full_text: str = ""
    sources: list[str] = Field(default_factory=list)   # symbol_ids included
    total_chars: int = 0
    truncated: bool = False


# ---------------------------------------------------------------------------
# Analysis Models
# ---------------------------------------------------------------------------


class ImpactAnalysisResult(BaseModel):
    """Result of a "what breaks if I change X" query."""

    target_symbol: str
    directly_affected: list[str] = Field(default_factory=list)
    transitively_affected: list[str] = Field(default_factory=list)
    affected_api_routes: list[str] = Field(default_factory=list)
    affected_files: list[str] = Field(default_factory=list)
    impact_depth: int = 0


class QueryResponse(BaseModel):
    """Final response returned to the user."""

    answer: str
    intent: QueryIntent
    sources: list[str] = Field(default_factory=list)
    context_chars: int = 0
    model_used: str = ""
