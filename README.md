# Coderr — Local-First AI Codebase Intelligence System

A real, working structural repository intelligence engine for Python codebases.
Built on AST parsing, graph-aware retrieval, semantic search, and hybrid RAG.
Runs fully locally using Ollama + Qdrant — no cloud, no API keys, no Docker.

---

## What It Is

Coderr converts a Python repository into structured knowledge:

- **AST-parsed symbols** — functions, methods, classes with exact line numbers
- **Global symbol registry** — cross-file call resolution (handles imports + aliases)
- **Dependency graph** — networkx DiGraph with IMPORTS/CALLS/DEFINES/INHERITS edges
- **Semantic embeddings** — BAAI/bge-small-en-v1.5 (384-dim), by function/class boundary
- **Hybrid retrieval** — 4-stage pipeline: semantic → keyword → graph expansion → reranking

## What It Is NOT

- Not a "chat with repo" tool
- Not a code autocomplete tool  
- Not a VSCode extension
- Not a generic RAG chatbot

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.ai) running locally with `qwen2.5:3b` pulled
- 2GB+ disk for embeddings model (downloaded automatically on first use)

---

## Installation

```bash
# Clone / navigate to the project
cd coderr

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env if needed (defaults work out of the box)

# Pull the Ollama model
ollama pull qwen2.5:3b
```

---

## Usage

### Step 1 — Index a Repository

```bash
python main.py index "D:/Projects/my_python_repo"
```

Or with a custom name:

```bash
python main.py index "D:/Projects/my_python_repo" --name my_repo
```

The system will:
1. Scan all `.py` files (ignores venv, __pycache__, .git, etc.)
2. Parse with AST — extract functions, classes, imports, calls
3. Build global symbol registry (cross-file call resolution)
4. Build networkx dependency/call graph
5. Embed all functions + classes with BAAI/bge-small-en-v1.5
6. Store in local Qdrant (no Docker, no server)

### Step 2 — Query the Repository

```bash
# Explain a flow
python main.py query "Explain the authentication flow"

# Find a symbol
python main.py query "Where is JWT validation implemented?"

# Impact analysis
python main.py query "What breaks if I modify refresh_access_token?"

# Dependencies
python main.py query "What does the login function depend on?"
```

### Step 3 — Structural Inspection

```bash
# Inspect a function (callers, callees, location)
python main.py inspect-function validate_token

# Get dependency chain
python main.py dependencies validate_token --depth 3

# Full impact analysis
python main.py impact validate_token
```

### Manage Repositories

```bash
# List indexed repos
python main.py list-repos

# Start the API server
python main.py serve --port 8000
```

---

## REST API

Start the server:

```bash
python main.py serve
```

Endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check + Ollama availability |
| GET | `/repos` | List indexed repositories |
| POST | `/index` | Index a repository |
| POST | `/query` | Natural language query |
| GET | `/dependencies/{repo}/{symbol}` | Dependency chain |
| GET | `/impact/{repo}/{symbol}` | Impact analysis |
| GET | `/inspect/{repo}/{fn_name}` | Function inspection |

API docs available at `http://localhost:8000/docs`

### Example API calls

```bash
# Index
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{"repo_path": "D:/Projects/my_repo"}'

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "explain authentication flow", "repo_name": "my_repo"}'

# Impact
curl http://localhost:8000/impact/my_repo/validate_token
```

---

## Configuration (.env)

```env
# Where Qdrant data and graph files are stored
CODERR_DATA_DIR=./coderr_data

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:3b
OLLAMA_TIMEOUT=120

# Supported models: qwen2.5:3b, mistral, phi3
# Switch per-query: python main.py query "..." --model mistral

# Retrieval tuning
MAX_RETRIEVAL_RESULTS=20
GRAPH_EXPANSION_DEPTH=2
MAX_CONTEXT_CHARS=8000

# Logging
LOG_LEVEL=INFO
```

---

## Architecture

```
Repository Path
     │
     ▼
Scanner (pathlib) → .py files only, ignores venv/__pycache__/.git/etc.
     │
     ▼
AST Parser (stdlib ast) → ParsedFile: functions, classes, imports, calls, line numbers
     │
     ├──────────────────────────────────────────────┐
     ▼                                              ▼
Symbol Registry                               Graph Engine (networkx)
(3-pass cross-file                            IMPORTS / CALLS / DEFINES / INHERITS edges
 call resolution)                             Callers populated on all function nodes
     │                                              │
     └──────────────────────────┬───────────────────┘
                                ▼
                     Ingestion Pipeline
                     (flat symbols → embed → Qdrant)
                                │
                     ┌──────────┴──────────┐
                     ▼                     ▼
                  Embedder            QdrantStore
             (bge-small-en-v1.5)   (local file mode)
             function/class only    no Docker needed
                                │
                     ┌──────────┘
                     ▼
              QUERY TIME:

              Intent Router (rule-based)
              → SYMBOL_LOOKUP / FLOW_EXPLANATION / IMPACT_ANALYSIS
                / DEPENDENCY_QUERY / GENERAL_QUERY

              4-Stage Retrieval Engine:
              Stage 1: Semantic (Qdrant vector search)
              Stage 2: Keyword/symbol boost (graph name match)
              Stage 3: Graph expansion (intent-specific traversal)
              Stage 4: Reranking (semantic + graph distance + intent alignment)

              Context Assembler (structured sections + budgeting):
              [ENTRY POINTS] → [CORE EXECUTION FLOW] → [DEPENDENCIES]
              → [RELATED UTILITIES] → [IMPORTANT IMPORTS]

              Ollama LLM (local, qwen2.5:3b default)
              ↓
              Grounded Answer
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| AST-only parsing | Reliable, zero regex, preserves line numbers |
| 3-pass symbol registry | Accurate cross-file call resolution including aliases |
| No file-level embeddings | Avoids noise — function/class boundaries are semantic units |
| Local Qdrant (no Docker) | File-mode client, zero infrastructure |
| 4-stage retrieval | Semantic alone misses structural relationships |
| Caller-before-callee ordering | Natural execution flow order for LLM context |
| Intent routing | Different query types need different traversal strategies |
| Impact analysis first-class | Reverse BFS is the most powerful structural feature |

---

## Running Tests

```bash
pytest app/tests/ -v
```

---

## Project Structure

```
coderr/
├── main.py                          # Entry point
├── requirements.txt
├── .env.example
└── app/
    ├── config/settings.py           # All configuration
    ├── schemas/models.py            # All Pydantic models
    ├── utils/logging_config.py      # Structured logging
    ├── parsing/
    │   ├── scanner.py               # File discovery
    │   ├── ast_parser.py            # AST extraction
    │   └── symbol_registry.py       # Cross-file symbol resolution
    ├── graph/graph_engine.py        # networkx dependency graph
    ├── embeddings/embedder.py       # bge-small-en-v1.5
    ├── vector_store/qdrant_store.py # Local Qdrant integration
    ├── ingestion/ingestion_pipeline.py
    ├── retrieval/
    │   ├── intent_router.py         # Rule-based query classification
    │   └── retrieval_engine.py      # 4-stage hybrid retrieval
    ├── reasoning/context_assembler.py
    ├── llm/ollama_client.py         # Ollama HTTP client
    ├── core/intelligence.py         # High-level orchestrator
    ├── cli/commands.py              # Typer CLI
    ├── api/server.py                # FastAPI
    └── tests/
        ├── test_ast_parser.py
        ├── test_symbol_registry.py
        ├── test_graph_engine.py
        ├── test_retrieval.py
        └── test_context_assembler.py
```
