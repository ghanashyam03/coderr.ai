from __future__ import annotations

import logging
import math
import re
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.schemas.models import CodeSymbol

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 100
_VECTOR_SIZE = 384


def _symbol_id_to_point_id(symbol_id: str) -> int:
    """Convert a string symbol_id to a deterministic non-negative integer suitable
    for use as a Qdrant point ID.

    We use Python's built-in ``hash()`` (which is stable within a single process
    but NOT across processes unless ``PYTHONHASHSEED`` is fixed).  To guarantee
    true cross-process stability we use a simple FNV-1a 64-bit hash instead.
    The result is masked to 63 bits so it always fits in a signed 64-bit integer.
    """
    # FNV-1a 64-bit — deterministic and process-independent
    FNV_PRIME: int = 0x100000001B3
    FNV_OFFSET: int = 0xCBF29CE484222325
    value = FNV_OFFSET
    for byte in symbol_id.encode("utf-8"):
        value ^= byte
        value = (value * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    # Keep within [0, 2^63 - 1]
    return value % (2**63 - 1)


class QdrantStore:
    """Local-file-mode Qdrant vector store for Coderr symbol embeddings.

    All data is persisted on disk at *data_dir*.  No Docker or external server
    is required — the ``qdrant_client`` library ships its own embedded storage
    backend.

    Usage::

        store = QdrantStore(data_dir="./coderr_data/qdrant")
        store.ensure_collection("my_repo")
        store.upsert_symbols("my_repo", symbols, vectors)
        results = store.search("my_repo", query_vector, top_k=10)
    """

    def __init__(self, data_dir: str) -> None:
        """Initialise the store and open (or create) the on-disk database.

        Args:
            data_dir: Filesystem path where Qdrant stores its segment files.
                      The directory is created automatically if it does not exist.
        """
        self._data_dir: str = data_dir
        self._vector_size: int = _VECTOR_SIZE

        logger.info("Opening QdrantClient at path=%r", data_dir)
        self._client: QdrantClient = QdrantClient(path=data_dir)
        logger.info("QdrantClient ready.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collection_name(self, repo_name: str) -> str:
        """Return a Qdrant-safe collection name derived from *repo_name*.

        Non-alphanumeric characters are replaced with underscores and the
        result is prefixed with ``coderr_`` to avoid accidental collisions
        with any existing Qdrant collections.

        Args:
            repo_name: Raw repository name (e.g. ``"my-awesome/repo.git"``).

        Returns:
            Sanitised collection name (e.g. ``"coderr_my_awesome_repo_git"``).
        """
        sanitised = re.sub(r"[^A-Za-z0-9]+", "_", repo_name).strip("_")
        return f"coderr_{sanitised}"

    def _build_payload(self, symbol: CodeSymbol) -> dict:
        """Build the Qdrant point payload from a :class:`CodeSymbol`.

        The full ``source`` field is excluded to keep the index compact; the
        first 500 characters are stored as ``source_preview`` instead.  The
        original ``symbol_id`` string is always included so callers can
        round-trip results back to the symbol without re-querying.

        Args:
            symbol: The code symbol to serialise.

        Returns:
            A plain dictionary safe to store in a Qdrant point payload.
        """
        data = symbol.model_dump(exclude={"source"})
        data["source_preview"] = (symbol.source or "")[:500]
        data["symbol_id"] = symbol.symbol_id
        data["full_source"] = True  # flag: full source lives on disk
        return data

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def ensure_collection(self, repo_name: str) -> None:
        """Create the vector collection for *repo_name* if it does not yet exist.

        If the collection is already present this method is a no-op — it will
        **not** truncate or modify the existing data.

        Args:
            repo_name: Repository name used to derive the collection name.
        """
        name = self._collection_name(repo_name)
        existing = {c.name for c in self._client.get_collections().collections}
        if name in existing:
            logger.debug("Collection %r already exists — skipping creation.", name)
            return

        logger.info("Creating collection %r (size=%d, distance=Cosine).", name, self._vector_size)
        self._client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=self._vector_size,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Collection %r created successfully.", name)

    def delete_collection(self, repo_name: str) -> None:
        """Permanently delete the collection for *repo_name*.

        Typically called before a full re-index so the old points do not
        persist alongside the freshly computed embeddings.

        Args:
            repo_name: Repository name whose collection should be removed.
        """
        name = self._collection_name(repo_name)
        logger.info("Deleting collection %r.", name)
        self._client.delete_collection(collection_name=name)
        logger.info("Collection %r deleted.", name)

    def collection_exists(self, repo_name: str) -> bool:
        """Return ``True`` when a collection for *repo_name* is present.

        Args:
            repo_name: Repository name to check.

        Returns:
            ``True`` if the collection exists, ``False`` otherwise.
        """
        name = self._collection_name(repo_name)
        existing = {c.name for c in self._client.get_collections().collections}
        return name in existing

    def get_collection_info(self, repo_name: str) -> dict:
        """Return metadata about the collection for *repo_name*.

        Args:
            repo_name: Repository name.

        Returns:
            Dictionary with at least the following keys:

            * ``collection_name`` — sanitised Qdrant name
            * ``points_count`` — number of indexed points
            * ``vector_size`` — dimensionality of stored vectors
            * ``distance`` — distance metric in use
            * ``status`` — collection status string

        Raises:
            ValueError: If the collection does not exist.
        """
        name = self._collection_name(repo_name)
        if not self.collection_exists(repo_name):
            raise ValueError(
                f"Collection for repo {repo_name!r} (Qdrant name: {name!r}) does not exist."
            )

        info = self._client.get_collection(collection_name=name)
        config = info.config
        vectors_config = config.params.vectors  # VectorParams or dict

        # vectors_config can be a VectorParams instance or a dict mapping name->VectorParams
        if isinstance(vectors_config, VectorParams):
            vec_size = vectors_config.size
            distance = str(vectors_config.distance)
        elif isinstance(vectors_config, dict):
            # Named vectors — pick the first (and usually only) entry
            first = next(iter(vectors_config.values()))
            vec_size = first.size
            distance = str(first.distance)
        else:
            vec_size = self._vector_size
            distance = "unknown"

        return {
            "collection_name": name,
            "points_count": info.points_count,
            "vector_size": vec_size,
            "distance": distance,
            "status": str(info.status),
        }

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def upsert_symbols(
        self,
        repo_name: str,
        symbols: list[CodeSymbol],
        vectors: list[list[float]],
    ) -> None:
        """Upsert symbol embeddings into the collection.

        Points are inserted (or updated if they already exist) in batches of
        100 to keep memory consumption predictable.  Each point carries:

        * A deterministic integer ID derived from ``symbol.symbol_id``.
        * The embedding vector.
        * A payload dict with all symbol metadata except the full source text
          (only the first 500 characters are stored as ``source_preview``).

        Args:
            repo_name: Repository name — used to select the target collection.
            symbols: List of :class:`CodeSymbol` objects to index.
            vectors: Corresponding embedding vectors (must be the same length
                     as *symbols* and each vector must have length 384).

        Raises:
            ValueError: If *symbols* and *vectors* have different lengths.
        """
        if len(symbols) != len(vectors):
            raise ValueError(
                f"symbols and vectors must have the same length "
                f"(got {len(symbols)} symbols and {len(vectors)} vectors)."
            )

        self.ensure_collection(repo_name)
        name = self._collection_name(repo_name)
        total = len(symbols)
        num_batches = math.ceil(total / _CHUNK_SIZE)

        logger.info(
            "Upserting %d symbols into collection %r in %d batch(es).",
            total,
            name,
            num_batches,
        )

        for batch_idx in range(num_batches):
            start = batch_idx * _CHUNK_SIZE
            end = min(start + _CHUNK_SIZE, total)
            batch_symbols = symbols[start:end]
            batch_vectors = vectors[start:end]

            points: list[PointStruct] = [
                PointStruct(
                    id=_symbol_id_to_point_id(sym.symbol_id),
                    vector=vec,
                    payload=self._build_payload(sym),
                )
                for sym, vec in zip(batch_symbols, batch_vectors)
            ]

            self._client.upsert(collection_name=name, points=points)
            logger.debug(
                "Batch %d/%d upserted (%d points, symbols[%d:%d]).",
                batch_idx + 1,
                num_batches,
                len(points),
                start,
                end,
            )

        logger.info("Upsert complete — %d symbols indexed in %r.", total, name)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def search(
        self,
        repo_name: str,
        query_vector: list[float],
        top_k: int = 20,
        symbol_type_filter: Optional[str] = None,
    ) -> list[dict]:
        """Perform a semantic nearest-neighbour search in the collection.

        Args:
            repo_name: Repository name — selects the target collection.
            query_vector: Embedding vector for the query (length must be 384).
            top_k: Maximum number of results to return.
            symbol_type_filter: If provided, only symbols whose ``symbol_type``
                matches this value are returned (e.g. ``"function"``).

        Returns:
            List of result dicts, ordered by descending similarity score.
            Each dict has:

            * ``symbol_id`` — the original string symbol ID.
            * ``score`` — cosine similarity score (higher is better).
            * ``payload`` — full metadata dictionary as stored in Qdrant.

        Raises:
            ValueError: If the collection for *repo_name* does not exist.
        """
        # Validate vector dimensions before querying
        if len(query_vector) != self._vector_size:
            raise ValueError(
                f"Query vector dimensions mismatch. Expected {self._vector_size}, got {len(query_vector)}."
            )

        # Validate collection existence before querying
        if not self.collection_exists(repo_name):
            raise ValueError(
                f"Collection for repo {repo_name!r} does not exist. "
                "Run ensure_collection() and upsert_symbols() first."
            )

        name = self._collection_name(repo_name)

        query_filter: Optional[Filter] = None
        if symbol_type_filter is not None:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="symbol_type",
                        match=MatchValue(value=symbol_type_filter),
                    )
                ]
            )

        logger.debug(
            "Searching collection %r — top_k=%d, filter=%r.",
            name,
            top_k,
            symbol_type_filter,
        )

        res = self._client.query_points(
            collection_name=name,
            query=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        hits = res.points


        results: list[dict] = []
        for hit in hits:
            payload: dict = hit.payload or {}
            symbol_id: str = payload.get("symbol_id", "")
            results.append(
                {
                    "symbol_id": symbol_id,
                    "score": float(hit.score),
                    "payload": payload,
                }
            )

        logger.debug("Search returned %d result(s) from %r.", len(results), name)
        return results

    def get_all_symbol_ids(self, repo_name: str) -> list[str]:
        """Retrieve every ``symbol_id`` stored in the collection.

        This method is used by the graph engine to cross-reference embedded
        symbols with the call/caller graph without loading full vectors.

        The implementation uses Qdrant's scroll API to page through all
        points efficiently regardless of collection size.

        Args:
            repo_name: Repository name whose symbols should be listed.

        Returns:
            List of ``symbol_id`` strings (order is not guaranteed).

        Raises:
            ValueError: If the collection for *repo_name* does not exist.
        """
        if not self.collection_exists(repo_name):
            raise ValueError(
                f"Collection for repo {repo_name!r} does not exist."
            )

        name = self._collection_name(repo_name)
        symbol_ids: list[str] = []
        offset = None  # Qdrant scroll cursor

        logger.debug("Scrolling all symbol_ids from collection %r.", name)

        while True:
            records, next_offset = self._client.scroll(
                collection_name=name,
                scroll_filter=None,
                limit=256,
                offset=offset,
                with_payload=["symbol_id"],
                with_vectors=False,
            )

            for record in records:
                if record.payload:
                    sid = record.payload.get("symbol_id")
                    if sid:
                        symbol_ids.append(sid)

            if next_offset is None:
                break
            offset = next_offset

        logger.debug(
            "Retrieved %d symbol_id(s) from collection %r.", len(symbol_ids), name
        )
        return symbol_ids
