from __future__ import annotations

"""
Embedding engine for Coderr.

Uses BAAI/bge-small-en-v1.5 (384-dimensional vectors) via sentence-transformers.
The SentenceTransformer model is loaded lazily on first use and cached as a
singleton for the lifetime of the process.

BGE document prefix : "Represent this code: "
BGE query prefix    : "Represent this question for searching relevant code: "
"""

import logging
from typing import Optional

from sentence_transformers import SentenceTransformer

from app.config.settings import settings
from app.schemas.models import CodeSymbol

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DOCUMENT_PREFIX: str = "Represent this code: "
_QUERY_PREFIX: str = "Represent this question for searching relevant code: "
_SOURCE_CHAR_LIMIT: int = 1_500
_VECTOR_SIZE: int = 384


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


class Embedder:
    """
    Sentence-transformer–based embedding engine for CodeSymbol objects.

    The underlying model is loaded *lazily* on the first call to :py:attr:`model`
    and then reused for the life of the process.  This avoids paying the
    multi-second startup cost unless embeddings are actually needed.
    """

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL) -> None:
        self._model_name: str = model_name
        self._model: Optional[SentenceTransformer] = None

    # ------------------------------------------------------------------
    # Model property (lazy singleton)
    # ------------------------------------------------------------------

    @property
    def model(self) -> SentenceTransformer:
        """Return the loaded SentenceTransformer, loading it on first access."""
        if self._model is None:
            logger.info(
                "Loading sentence-transformer model '%s' — this may take a moment …",
                self._model_name,
            )
            self._model = SentenceTransformer(self._model_name)
            logger.info(
                "Model '%s' loaded successfully (vector size: %d).",
                self._model_name,
                _VECTOR_SIZE,
            )
        return self._model

    # ------------------------------------------------------------------
    # Text construction
    # ------------------------------------------------------------------

    def _build_embedding_text(self, symbol: CodeSymbol) -> str:
        """
        Construct the document text that will be embedded for *symbol*.

        Layout::

            {symbol_type}: {qualified_name}
            {docstring}          ← omitted when absent
            {source[:1500]}      ← first 1 500 chars to stay within model limits

        The BGE document prefix is prepended before encoding, not here —
        that way the raw text can be inspected / logged independently.

        If ``source`` is empty or blank, ``qualified_name`` is used as the
        fallback body so the embedding is still meaningful.
        """
        parts: list[str] = []

        # Header line
        parts.append(f"{symbol.symbol_type.value}: {symbol.qualified_name}")

        # Optional docstring
        if symbol.docstring and symbol.docstring.strip():
            parts.append(symbol.docstring.strip())

        # Source body — truncated to avoid tokeniser overflow
        body = (symbol.source or "").strip()
        if not body:
            # Fallback: use the qualified name so the vector is still informative
            body = symbol.qualified_name
        parts.append(body[:_SOURCE_CHAR_LIMIT])

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Public encoding API
    # ------------------------------------------------------------------

    def embed_symbol(self, symbol: CodeSymbol) -> list[float]:
        """
        Embed a single :class:`~app.schemas.models.CodeSymbol`.

        Returns a normalised 384-dimensional vector as a plain Python
        ``list[float]`` (JSON-serialisable).
        """
        text = _DOCUMENT_PREFIX + self._build_embedding_text(symbol)
        vector = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector.tolist()

    def embed_symbols_batch(
        self,
        symbols: list[CodeSymbol],
        batch_size: int = settings.EMBEDDING_BATCH_SIZE,
    ) -> list[list[float]]:
        """
        Embed a list of :class:`~app.schemas.models.CodeSymbol` objects in
        batches for efficiency.

        Progress is logged every 100 symbols so long-running indexing jobs
        remain observable.

        Returns vectors in the *same order* as *symbols*.
        """
        if not symbols:
            return []

        texts: list[str] = [
            _DOCUMENT_PREFIX + self._build_embedding_text(s) for s in symbols
        ]

        results: list[list[float]] = []
        total = len(texts)

        for start in range(0, total, batch_size):
            batch_texts = texts[start : start + batch_size]
            vectors = self.model.encode(
                batch_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            results.extend(v.tolist() for v in vectors)

            processed = min(start + batch_size, total)
            # Log every 100 symbols (and always log the final batch)
            if processed % 100 == 0 or processed == total:
                logger.info(
                    "Embedding progress: %d / %d symbols encoded.",
                    processed,
                    total,
                )

        return results

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a free-text search *query* using the BGE query prefix.

        Returns a normalised 384-dimensional vector as ``list[float]``.
        """
        prefixed = _QUERY_PREFIX + query.strip()
        vector = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vector.tolist()

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_vector_size(self) -> int:
        """Return the fixed output dimensionality of this model (384)."""
        return _VECTOR_SIZE
