"""Read tools for accessing vault documents."""

import logging
from pathlib import Path

from ..db.sqlite import SQLiteDB
from ..parser.markdown import MarkdownParser

logger = logging.getLogger(__name__)


class ReadEngine:
    """Engine for reading and navigating vault documents."""

    def __init__(
        self,
        vault_path: Path,
        sqlite_db: SQLiteDB,
        parser: MarkdownParser,
    ):
        """Initialize the read engine.

        Args:
            vault_path: Path to the vault directory.
            sqlite_db: SQLite database instance.
            parser: Markdown parser instance.
        """
        self.vault_path = vault_path
        self.sqlite_db = sqlite_db
        self.parser = parser

    def read_note(
        self,
        filename: str,
        include_backlinks: bool = True,
        include_related: bool = True,
    ) -> dict:
        """Read a note with its metadata and relationships.

        Args:
            filename: Note filename (with or without .md extension).
            include_backlinks: Whether to include backlink information.
            include_related: Whether to include related document summaries.

        Returns:
            Note content and metadata.
        """
        # Normalize filename
        if not filename.endswith(".md"):
            filename = f"{filename}.md"

        # Try to find the file
        file_path = self._resolve_filename(filename)
        if not file_path:
            return {
                "success": False,
                "error": f"Note not found: {filename}",
            }

        # Parse the document
        try:
            doc = self.parser.parse_file(file_path)
        except Exception as e:
            logger.error(f"Error parsing {filename}: {e}")
            return {
                "success": False,
                "error": f"Error parsing note: {str(e)}",
            }

        result = {
            "success": True,
            "id": doc.id,
            "title": doc.title,
            "type": doc.type,
            "status": doc.status,
            "tags": doc.tags,
            "file_path": str(file_path),
            "content": doc.body,
            "metadata": doc.raw_metadata,
        }

        # Add backlink information
        if include_backlinks:
            # Outgoing links (this document links to)
            outgoing_links = []
            for bl in doc.backlinks:
                target_path = self.parser.resolve_link(bl.target)
                outgoing_links.append(
                    {
                        "target": bl.target,
                        "context": bl.context,
                        "exists": target_path is not None,
                    }
                )
            result["outgoing_links"] = outgoing_links

            # Incoming links (documents that link to this)
            incoming_edges = self.sqlite_db.get_incoming_edges(doc.id)
            incoming_links = []
            for edge in incoming_edges:
                source_node = self.sqlite_db.get_node(edge.source_id)
                if source_node:
                    incoming_links.append(
                        {
                            "source_id": edge.source_id,
                            "source_title": source_node.title,
                            "context": edge.context_snippet,
                        }
                    )
            result["incoming_links"] = incoming_links

        # Add related document summaries
        if include_related:
            related = []
            seen_ids = {doc.id}

            # Add documents from outgoing links
            for bl in doc.backlinks[:5]:  # Limit to 5
                target_path = self.parser.resolve_link(bl.target)
                if target_path:
                    target_doc = self.parser.parse_file(target_path)
                    if target_doc.id not in seen_ids:
                        seen_ids.add(target_doc.id)
                        related.append(
                            {
                                "id": target_doc.id,
                                "title": target_doc.title,
                                "type": target_doc.type,
                                "summary": target_doc.get_summary(150),
                                "relationship": "linked_to",
                            }
                        )

            # Add documents from incoming links
            for edge in incoming_edges[:5]:
                if edge.source_id not in seen_ids:
                    source_node = self.sqlite_db.get_node(edge.source_id)
                    if source_node:
                        seen_ids.add(edge.source_id)
                        related.append(
                            {
                                "id": source_node.id,
                                "title": source_node.title,
                                "type": source_node.type,
                                "summary": source_node.summary,
                                "relationship": "linked_from",
                            }
                        )

            result["related_documents"] = related

        return result

    def _resolve_filename(self, filename: str) -> Path | None:
        """Resolve a filename to a file path.

        Args:
            filename: Filename to resolve.

        Returns:
            Resolved path or None.
        """
        # Try direct path
        direct_path = self.vault_path / filename
        if direct_path.exists():
            return direct_path

        # Try without extension
        stem = Path(filename).stem
        for md_file in self.vault_path.rglob("*.md"):
            if md_file.stem.lower() == stem.lower():
                return md_file

        # Try by ID in database
        node = self.sqlite_db.get_node(stem)
        if node:
            path = Path(node.file_path)
            if path.exists():
                return path

        return None

    def get_note_by_id(self, doc_id: str) -> dict:
        """Get a note by its ID.

        Args:
            doc_id: Document ID.

        Returns:
            Note content and metadata.
        """
        node = self.sqlite_db.get_node(doc_id)
        if not node:
            return {
                "success": False,
                "error": f"Note not found with ID: {doc_id}",
            }

        return self.read_note(Path(node.file_path).name)

    def list_notes(
        self,
        type_filter: str | None = None,
        tag_filter: str | None = None,
        limit: int = 50,
    ) -> list[dict]:
        """List notes with optional filters.

        Args:
            type_filter: Filter by document type.
            tag_filter: Filter by tag.
            limit: Maximum number of results.

        Returns:
            List of note summaries.
        """
        if type_filter:
            nodes = self.sqlite_db.get_nodes_by_type(type_filter)
        else:
            nodes = self.sqlite_db.get_all_nodes()

        results = []
        for node in nodes[:limit]:
            # Apply tag filter
            if tag_filter and (not node.tags or tag_filter not in node.tags):
                continue

            results.append(
                {
                    "id": node.id,
                    "title": node.title,
                    "type": node.type,
                    "status": node.status,
                    "tags": node.tags,
                    "summary": node.summary,
                    "updated_at": node.updated_at.isoformat() if node.updated_at else None,
                }
            )

        return results

    def get_note_graph(self, doc_id: str, depth: int = 1) -> dict:
        """Get the graph context around a note.

        Args:
            doc_id: Central document ID.
            depth: How many hops to include.

        Returns:
            Graph structure with nodes and edges.
        """
        visited: set[str] = set()
        nodes_data: list[dict] = []
        edges_data: list[dict] = []

        def explore(current_id: str, current_depth: int):
            if current_id in visited or current_depth > depth:
                return
            visited.add(current_id)

            node = self.sqlite_db.get_node(current_id)
            if not node:
                return

            nodes_data.append(
                {
                    "id": node.id,
                    "title": node.title,
                    "type": node.type,
                    "is_center": current_id == doc_id,
                }
            )

            # Get outgoing edges
            outgoing = self.sqlite_db.get_outgoing_edges(current_id)
            for edge in outgoing:
                edges_data.append(
                    {
                        "source": edge.source_id,
                        "target": edge.target_id,
                        "context": edge.context_snippet,
                    }
                )
                explore(edge.target_id, current_depth + 1)

            # Get incoming edges
            incoming = self.sqlite_db.get_incoming_edges(current_id)
            for edge in incoming:
                if edge.source_id not in visited:
                    edges_data.append(
                        {
                            "source": edge.source_id,
                            "target": edge.target_id,
                            "context": edge.context_snippet,
                        }
                    )
                explore(edge.source_id, current_depth + 1)

        explore(doc_id, 0)

        return {
            "center": doc_id,
            "depth": depth,
            "nodes": nodes_data,
            "edges": edges_data,
        }
