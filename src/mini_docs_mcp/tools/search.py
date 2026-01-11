"""Search tools for the vault."""

import logging

from ..db.sqlite import SQLiteDB
from ..db.vector import VectorDB

logger = logging.getLogger(__name__)


class SearchEngine:
    """Hybrid search engine combining semantic and metadata search."""

    def __init__(self, sqlite_db: SQLiteDB, vector_db: VectorDB):
        """Initialize the search engine.

        Args:
            sqlite_db: SQLite database instance.
            vector_db: Vector database instance.
        """
        self.sqlite_db = sqlite_db
        self.vector_db = vector_db

    def search(
        self,
        query: str,
        type_filter: str | None = None,
        tag_filter: list[str] | None = None,
        top_k: int = 10,
        include_graph_context: bool = True,
    ) -> dict:
        """Perform hybrid search.

        Args:
            query: Search query string.
            type_filter: Filter by document type.
            tag_filter: Filter by tags.
            top_k: Number of results to return.
            include_graph_context: Whether to include backlink context.

        Returns:
            Search results with metadata and scores.
        """
        # Step 1: Semantic search via vector DB
        vector_results = self.vector_db.search(query, top_k=top_k * 2)

        # Step 2: Get unique document IDs
        doc_scores: dict[str, float] = {}
        doc_chunks: dict[str, list[dict]] = {}

        for result in vector_results:
            doc_id = result["doc_id"]
            score = 1.0 - result["score"]  # Convert distance to similarity

            if doc_id not in doc_scores:
                doc_scores[doc_id] = score
                doc_chunks[doc_id] = []
            else:
                # Aggregate scores for same document
                doc_scores[doc_id] = max(doc_scores[doc_id], score)

            doc_chunks[doc_id].append(
                {
                    "content": result["content"],
                    "score": score,
                }
            )

        # Step 3: Enrich with metadata and apply filters
        enriched_results = []
        for doc_id, score in doc_scores.items():
            node = self.sqlite_db.get_node(doc_id)
            if not node:
                continue

            # Apply type filter
            if type_filter and node.type != type_filter:
                continue

            # Apply tag filter
            if tag_filter and node.tags:
                if not any(tag in node.tags for tag in tag_filter):
                    continue
            elif tag_filter and not node.tags:
                continue

            # Calculate combined score (semantic + reference weight)
            incoming_edges = self.sqlite_db.get_incoming_edges(doc_id)
            ref_weight = len(incoming_edges) * 0.1  # Bonus for referenced docs
            combined_score = score + min(ref_weight, 0.3)  # Cap at 0.3 bonus

            result_entry = {
                "id": doc_id,
                "title": node.title,
                "type": node.type,
                "status": node.status,
                "tags": node.tags,
                "file_path": node.file_path,
                "summary": node.summary,
                "score": combined_score,
                "semantic_score": score,
                "reference_count": len(incoming_edges),
                "matching_chunks": doc_chunks.get(doc_id, [])[:3],  # Top 3 chunks
            }

            # Add graph context if requested
            if include_graph_context:
                outgoing_edges = self.sqlite_db.get_outgoing_edges(doc_id)
                result_entry["links_to"] = [e.target_id for e in outgoing_edges]
                result_entry["linked_from"] = [e.source_id for e in incoming_edges]

            enriched_results.append(result_entry)

        # Sort by combined score
        enriched_results.sort(key=lambda x: x["score"], reverse=True)

        return {
            "query": query,
            "total_results": len(enriched_results),
            "results": enriched_results[:top_k],
            "filters_applied": {
                "type": type_filter,
                "tags": tag_filter,
            },
        }

    def search_with_expansion(
        self,
        query: str,
        top_k: int = 5,
        expansion_hops: int = 1,
    ) -> dict:
        """Search with graph expansion (2-hop context).

        Args:
            query: Search query string.
            top_k: Number of initial results.
            expansion_hops: Number of hops to expand.

        Returns:
            Search results with expanded context.
        """
        # Initial search
        initial_results = self.search(query, top_k=top_k, include_graph_context=True)

        if expansion_hops == 0 or not initial_results["results"]:
            return initial_results

        # Collect connected documents
        primary_ids = set(r["id"] for r in initial_results["results"])
        expanded_ids: set[str] = set()

        for result in initial_results["results"]:
            # Add outgoing links
            for target_id in result.get("links_to", []):
                if target_id not in primary_ids:
                    expanded_ids.add(target_id)

            # Add incoming links
            for source_id in result.get("linked_from", []):
                if source_id not in primary_ids:
                    expanded_ids.add(source_id)

        # Get metadata for expanded nodes
        expanded_context = []
        for doc_id in expanded_ids:
            node = self.sqlite_db.get_node(doc_id)
            if node:
                expanded_context.append(
                    {
                        "id": doc_id,
                        "title": node.title,
                        "type": node.type,
                        "summary": node.summary,
                        "relationship": self._get_relationship(doc_id, primary_ids),
                    }
                )

        initial_results["expanded_context"] = expanded_context
        initial_results["expansion_hops"] = expansion_hops

        return initial_results

    def _get_relationship(self, doc_id: str, primary_ids: set[str]) -> str:
        """Determine relationship of a document to primary results."""
        incoming = self.sqlite_db.get_incoming_edges(doc_id)
        outgoing = self.sqlite_db.get_outgoing_edges(doc_id)

        relationships = []
        for edge in incoming:
            if edge.source_id in primary_ids:
                relationships.append(f"referenced by {edge.source_id}")
        for edge in outgoing:
            if edge.target_id in primary_ids:
                relationships.append(f"references {edge.target_id}")

        return "; ".join(relationships) if relationships else "connected"

    def find_similar(self, doc_id: str, top_k: int = 5) -> list[dict]:
        """Find documents similar to a given document.

        Args:
            doc_id: Document ID to find similar documents for.
            top_k: Number of results.

        Returns:
            List of similar documents.
        """
        # Get the document's content
        chunks = self.vector_db.get_document_chunks(doc_id)
        if not chunks:
            return []

        # Use the first chunk as representative
        representative_text = chunks[0]["content"]

        # Search excluding the source document
        results = self.vector_db.search(representative_text, top_k=top_k + 1)

        # Filter out the source document and enrich
        similar = []
        for result in results:
            if result["doc_id"] == doc_id:
                continue

            node = self.sqlite_db.get_node(result["doc_id"])
            if node:
                similar.append(
                    {
                        "id": result["doc_id"],
                        "title": node.title,
                        "type": node.type,
                        "similarity_score": 1.0 - result["score"],
                    }
                )

            if len(similar) >= top_k:
                break

        return similar
