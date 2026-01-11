"""LanceDB vector database operations for semantic search."""

import logging
from collections.abc import Callable
from pathlib import Path

import lancedb
from lancedb.pydantic import LanceModel, Vector

logger = logging.getLogger(__name__)

# Default embedding dimension (for sentence-transformers all-MiniLM-L6-v2)
DEFAULT_EMBEDDING_DIM = 384


class DocumentChunk(LanceModel):
    """Schema for document chunks in LanceDB."""

    chunk_id: str  # Unique ID for the chunk
    doc_id: str  # Parent document ID
    file_path: str  # Path to the source file
    content: str  # Chunk text content
    chunk_index: int  # Position in the document
    vector: Vector(DEFAULT_EMBEDDING_DIM)  # type: ignore


class VectorDB:
    """LanceDB vector database manager."""

    TABLE_NAME = "document_chunks"

    def __init__(
        self,
        db_path: Path | str,
        embedding_fn: Callable[[list[str]], list[list[float]]] | None = None,
    ):
        """Initialize the vector database.

        Args:
            db_path: Path to the LanceDB directory.
            embedding_fn: Function to generate embeddings from text.
                         Should accept a list of strings and return list of vectors.
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.db_path))
        self._embedding_fn = embedding_fn
        self._table = None
        self._init_table()

    def _init_table(self) -> None:
        """Initialize the document chunks table."""
        try:
            self._table = self.db.open_table(self.TABLE_NAME)
            logger.info(f"Opened existing table: {self.TABLE_NAME}")
        except Exception:
            # Table doesn't exist, will be created on first insert
            self._table = None
            logger.info(f"Table {self.TABLE_NAME} will be created on first insert")

    def set_embedding_function(
        self, embedding_fn: Callable[[list[str]], list[list[float]]]
    ) -> None:
        """Set the embedding function.

        Args:
            embedding_fn: Function to generate embeddings.
        """
        self._embedding_fn = embedding_fn

    def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for a list of texts.

        Args:
            texts: List of text strings.

        Returns:
            List of embedding vectors.
        """
        if self._embedding_fn is None:
            raise ValueError("Embedding function not set. Call set_embedding_function first.")
        return self._embedding_fn(texts)

    def add_document(
        self,
        doc_id: str,
        file_path: str,
        chunks: list[str],
    ) -> int:
        """Add a document's chunks to the vector database.

        Args:
            doc_id: Document ID.
            file_path: Path to the source file.
            chunks: List of text chunks from the document.

        Returns:
            Number of chunks added.
        """
        if not chunks:
            return 0

        # Generate embeddings
        embeddings = self._get_embeddings(chunks)

        # Create chunk records
        records = []
        for i, (chunk, vector) in enumerate(zip(chunks, embeddings, strict=True)):
            records.append(
                {
                    "chunk_id": f"{doc_id}_{i}",
                    "doc_id": doc_id,
                    "file_path": file_path,
                    "content": chunk,
                    "chunk_index": i,
                    "vector": vector,
                }
            )

        # Insert into table
        if self._table is None:
            self._table = self.db.create_table(self.TABLE_NAME, records)
            logger.info(f"Created table {self.TABLE_NAME} with {len(records)} records")
        else:
            self._table.add(records)
            logger.debug(f"Added {len(records)} chunks for document {doc_id}")

        return len(records)

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks for a document.

        Args:
            doc_id: Document ID to delete.

        Returns:
            Number of chunks deleted.
        """
        if self._table is None:
            return 0

        # Count before deletion
        try:
            count_before = self._table.count_rows(f"doc_id = '{doc_id}'")
        except Exception:
            count_before = 0

        if count_before > 0:
            self._table.delete(f"doc_id = '{doc_id}'")
            logger.debug(f"Deleted {count_before} chunks for document {doc_id}")

        return count_before

    def update_document(
        self,
        doc_id: str,
        file_path: str,
        chunks: list[str],
    ) -> int:
        """Update a document by deleting old chunks and adding new ones.

        Args:
            doc_id: Document ID.
            file_path: Path to the source file.
            chunks: New list of text chunks.

        Returns:
            Number of new chunks added.
        """
        self.delete_document(doc_id)
        return self.add_document(doc_id, file_path, chunks)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_expr: str | None = None,
    ) -> list[dict]:
        """Search for similar documents.

        Args:
            query: Search query text.
            top_k: Number of results to return.
            filter_expr: Optional SQL filter expression.

        Returns:
            List of search results with scores.
        """
        if self._table is None:
            return []

        # Generate query embedding
        query_embedding = self._get_embeddings([query])[0]

        # Build search query
        search_query = self._table.search(query_embedding).limit(top_k)

        if filter_expr:
            search_query = search_query.where(filter_expr)

        # Execute search
        results = search_query.to_list()

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "chunk_id": result["chunk_id"],
                    "doc_id": result["doc_id"],
                    "file_path": result["file_path"],
                    "content": result["content"],
                    "chunk_index": result["chunk_index"],
                    "score": float(result.get("_distance", 0)),
                }
            )

        return formatted_results

    def search_by_doc_ids(
        self,
        query: str,
        doc_ids: list[str],
        top_k: int = 10,
    ) -> list[dict]:
        """Search within specific documents.

        Args:
            query: Search query text.
            doc_ids: List of document IDs to search within.
            top_k: Number of results to return.

        Returns:
            List of search results.
        """
        if not doc_ids:
            return []

        # Build filter expression
        ids_str = ", ".join(f"'{id}'" for id in doc_ids)
        filter_expr = f"doc_id IN ({ids_str})"

        return self.search(query, top_k, filter_expr)

    def get_document_chunks(self, doc_id: str) -> list[dict]:
        """Get all chunks for a document.

        Args:
            doc_id: Document ID.

        Returns:
            List of chunk records.
        """
        if self._table is None:
            return []

        results = self._table.search().where(f"doc_id = '{doc_id}'").limit(1000).to_list()
        return sorted(results, key=lambda x: x["chunk_index"])

    def get_stats(self) -> dict:
        """Get vector database statistics.

        Returns:
            Dictionary with stats.
        """
        if self._table is None:
            return {
                "total_chunks": 0,
                "unique_documents": 0,
            }

        try:
            total_chunks = self._table.count_rows()
            # Get unique doc_ids
            all_chunks = self._table.search().limit(10000).to_list()
            unique_docs = len(set(c["doc_id"] for c in all_chunks))

            return {
                "total_chunks": total_chunks,
                "unique_documents": unique_docs,
            }
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {
                "total_chunks": 0,
                "unique_documents": 0,
            }


def create_default_embedding_function() -> Callable[[list[str]], list[list[float]]]:
    """Create default embedding function using sentence-transformers.

    Returns:
        Embedding function.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(texts: list[str]) -> list[list[float]]:
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    return embed


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: Text to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters.

    Returns:
        List of text chunks.
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size

        # Try to break at a paragraph or sentence boundary
        if end < text_len:
            # Look for paragraph break
            para_break = text.rfind("\n\n", start, end)
            if para_break > start + chunk_size // 2:
                end = para_break + 2
            else:
                # Look for sentence break
                for sep in [". ", "! ", "? ", "\n"]:
                    sent_break = text.rfind(sep, start, end)
                    if sent_break > start + chunk_size // 2:
                        end = sent_break + len(sep)
                        break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - chunk_overlap
        if start >= text_len:
            break

    return chunks
