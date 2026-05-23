from __future__ import annotations

"""
Integration Tests for QdrantStore and Semantic Retrieval.

Tests cover:
  1. Collection creation and verification
  2. deterministic point ID mapping
  3. Upserting symbol embeddings and querying via client.query_points()
  4. Query vector dimension validation (expects ValueError on mismatch)
  5. Collection existence validation before queries (expects ValueError)
  6. RetrievalEngine semantic stage error handling and graceful fallback
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock

from qdrant_client.models import Distance

from app.vector_store.qdrant_store import QdrantStore, _symbol_id_to_point_id
from app.retrieval.retrieval_engine import RetrievalEngine
from app.schemas.models import CodeSymbol, SymbolType, RetrievalResult


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_qdrant_store(tmp_path: Path) -> QdrantStore:
    """Fixture providing a fresh local file-based QdrantStore."""
    data_dir = tmp_path / "qdrant_db"
    return QdrantStore(data_dir=str(data_dir))


def _make_dummy_symbol(symbol_id: str, name: str, qualified_name: str) -> CodeSymbol:
    """Create a dummy CodeSymbol for vector store indexing."""
    return CodeSymbol(
        symbol_id=symbol_id,
        symbol_type=SymbolType.FUNCTION,
        name=name,
        qualified_name=qualified_name,
        file_path="/repo/auth/jwt.py",
        line_start=10,
        line_end=20,
        source="def dummy(): pass",
        docstring="Docstring",
        class_name=None,
        repo_name="dummy_repo",
        calls=[],
        callers=[],
        bases=[],
        decorators=[],
    )


# ---------------------------------------------------------------------------
# Point ID Mapping Tests
# ---------------------------------------------------------------------------

class TestPointIDMapping:
    def test_point_id_mapping_is_deterministic(self) -> None:
        """Verify that FNV-1a point ID mapping is deterministic and cross-process stable."""
        sid = "auth/jwt.py::validate_token"
        id1 = _symbol_id_to_point_id(sid)
        id2 = _symbol_id_to_point_id(sid)
        assert id1 == id2
        assert 0 <= id1 < 2**63 - 1

    def test_different_ids_produce_different_hashes(self) -> None:
        """Verify distinct symbol IDs produce different integers."""
        id1 = _symbol_id_to_point_id("auth/jwt.py::login")
        id2 = _symbol_id_to_point_id("auth/jwt.py::logout")
        assert id1 != id2


# ---------------------------------------------------------------------------
# Qdrant Store Integration Tests
# ---------------------------------------------------------------------------

class TestQdrantStoreSemanticQuery:
    def test_ensure_and_delete_collection(self, temp_qdrant_store: QdrantStore) -> None:
        """Verify collection lifecycle management."""
        repo = "my_repo"
        assert not temp_qdrant_store.collection_exists(repo)

        # Create
        temp_qdrant_store.ensure_collection(repo)
        assert temp_qdrant_store.collection_exists(repo)

        # Info
        info = temp_qdrant_store.get_collection_info(repo)
        assert "coderr_my_repo" in info["collection_name"]
        assert info["vector_size"] == 384
        assert "COSINE" in info["distance"].upper()

        # Delete
        temp_qdrant_store.delete_collection(repo)
        assert not temp_qdrant_store.collection_exists(repo)

    def test_upsert_and_search_success(self, temp_qdrant_store: QdrantStore) -> None:
        """Verify upserting and retrieval of symbols using QdrantLocal modern query APIs."""
        repo = "test_repo"
        temp_qdrant_store.ensure_collection(repo)

        # 384-dimensional vector
        vec_alpha = [0.1] * 384
        vec_beta = [0.9] * 384

        sym_alpha = _make_dummy_symbol("id_alpha", "alpha", "mod.alpha")
        sym_beta = _make_dummy_symbol("id_beta", "beta", "mod.beta")

        # Upsert
        temp_qdrant_store.upsert_symbols(repo, [sym_alpha, sym_beta], [vec_alpha, vec_beta])

        # Query semantic search matching beta
        query_vec = [0.89] * 384
        hits = temp_qdrant_store.search(repo, query_vec, top_k=5)

        assert len(hits) == 2
        # Beta should have a higher Cosine similarity score than Alpha
        assert hits[0]["symbol_id"] == "id_beta"
        assert hits[0]["score"] > 0.99
        assert hits[0]["payload"]["name"] == "beta"
        assert hits[0]["payload"]["source_preview"] == "def dummy(): pass"

        # Query all symbol IDs scroll API
        all_ids = temp_qdrant_store.get_all_symbol_ids(repo)
        assert set(all_ids) == {"id_alpha", "id_beta"}

    def test_search_validates_dimension(self, temp_qdrant_store: QdrantStore) -> None:
        """Verify ValueError is raised if query vector dimension does not match index size (384)."""
        repo = "test_repo"
        temp_qdrant_store.ensure_collection(repo)

        invalid_vec = [0.1] * 128  # Dimension is 128, expected 384
        with pytest.raises(ValueError) as excinfo:
            temp_qdrant_store.search(repo, invalid_vec, top_k=2)

        assert "Query vector dimensions mismatch" in str(excinfo.value)

    def test_search_validates_collection_existence(self, temp_qdrant_store: QdrantStore) -> None:
        """Verify ValueError is raised if querying a non-existent repository collection."""
        repo = "non_existent_repo"
        query_vec = [0.1] * 384

        with pytest.raises(ValueError) as excinfo:
            temp_qdrant_store.search(repo, query_vec, top_k=2)

        assert "does not exist" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Retrieval Engine Semantic Stage & Fallback Tests
# ---------------------------------------------------------------------------

class TestRetrievalFallback:
    def test_graceful_fallback_on_semantic_failure(self) -> None:
        """Verify that RetrievalEngine handles Qdrant search failures gracefully and falls back."""
        mock_graph = MagicMock()
        mock_embedder = MagicMock()
        mock_qdrant = MagicMock()

        # Simulate a database connection failure or general exception in QdrantStore search
        mock_qdrant.search.side_effect = RuntimeError("Qdrant database is disconnected")

        # Mock query embed to return a valid vector list
        mock_embedder.embed_query.return_value = [0.1] * 384

        # Mock keyword search in graph to return a fallback symbol
        mock_graph.find_symbol_nodes.return_value = ["func::mod.fallback"]
        mock_graph.get_node_data.return_value = {
            "node_type": "function",
            "symbol_type": "function",
            "name": "fallback",
            "qualified_name": "mod.fallback",
            "file_path": "/repo/auth/jwt.py",
            "decorators": [],
        }

        engine = RetrievalEngine(
            graph=mock_graph,
            embedder=mock_embedder,
            qdrant_store=mock_qdrant,
            repo_name="my_repo",
            max_results=5
        )

        # Execute retrieval pipeline
        results = engine.retrieve("explain the fallback logic")

        # Even though semantic search crashed, the engine should not crash,
        # logging the exception and falling back gracefully to keyword/graph matches!
        assert len(results) >= 1
        assert results[0].symbol.name == "fallback"
        assert results[0].source == "keyword"
        mock_qdrant.search.assert_called_once()
