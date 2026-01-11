"""Link tools for managing backlinks between documents."""

import logging
from pathlib import Path

from mini_docs_mcp.db.sqlite import Edge, SQLiteDB
from mini_docs_mcp.parser.markdown import MarkdownParser
from mini_docs_mcp.tools.write import WriteEngine

logger = logging.getLogger(__name__)


class LinkEngine:
    """Engine for managing links between documents."""

    def __init__(
        self,
        vault_path: Path,
        sqlite_db: SQLiteDB,
        parser: MarkdownParser,
        write_engine: WriteEngine,
    ):
        """Initialize the link engine.

        Args:
            vault_path: Path to the vault directory.
            sqlite_db: SQLite database instance.
            parser: Markdown parser instance.
            write_engine: Write engine for modifying files.
        """
        self.vault_path = vault_path
        self.sqlite_db = sqlite_db
        self.parser = parser
        self.write_engine = write_engine

    def link_notes(
        self,
        source: str,
        target: str,
        reason: str | None = None,
        bidirectional: bool = False,
    ) -> dict:
        """Create a backlink between two notes.

        Args:
            source: Source note filename or ID.
            target: Target note filename or ID.
            reason: Optional context/reason for the link.
            bidirectional: Whether to add link in both directions.

        Returns:
            Operation result.
        """
        # Resolve source file
        source_path = self._resolve_note(source)
        if not source_path:
            return {
                "success": False,
                "error": f"Source note not found: {source}",
            }

        # Resolve target file
        target_path = self._resolve_note(target)
        if not target_path:
            return {
                "success": False,
                "error": f"Target note not found: {target}",
            }

        # Parse source to get its content
        source_doc = self.parser.parse_file(source_path)
        target_doc = self.parser.parse_file(target_path)

        # Check if link already exists
        for bl in source_doc.backlinks:
            if (
                bl.target.lower() == target_doc.id.lower()
                or bl.target.lower() == target_path.stem.lower()
            ):
                return {
                    "success": False,
                    "error": "Link already exists",
                    "existing_context": bl.context,
                }

        # Read source content
        source_content = source_path.read_text(encoding="utf-8")

        # Add backlink to source
        updated_content = self.parser.add_backlink(source_content, target_doc.id, reason)

        # Write updated source
        source_path.write_text(updated_content, encoding="utf-8")
        logger.info(f"Added link: {source_doc.id} -> {target_doc.id}")

        # Update database
        edge = Edge(
            id=None,
            source_id=source_doc.id,
            target_id=target_doc.id,
            context_snippet=reason,
        )
        self.sqlite_db.upsert_edge(edge)

        results = {
            "success": True,
            "action": "linked",
            "source": source_doc.id,
            "target": target_doc.id,
            "reason": reason,
        }

        # Handle bidirectional
        if bidirectional:
            back_result = self.link_notes(target_doc.id, source_doc.id, reason, bidirectional=False)
            results["bidirectional"] = back_result.get("success", False)

        return results

    def unlink_notes(self, source: str, target: str) -> dict:
        """Remove a backlink between two notes.

        Args:
            source: Source note filename or ID.
            target: Target note filename or ID.

        Returns:
            Operation result.
        """
        # Resolve source file
        source_path = self._resolve_note(source)
        if not source_path:
            return {
                "success": False,
                "error": f"Source note not found: {source}",
            }

        # Parse notes
        source_doc = self.parser.parse_file(source_path)

        # Find and remove the link
        source_content = source_path.read_text(encoding="utf-8")
        import re

        # Try different link patterns
        patterns = [
            rf"\[\[{re.escape(target)}\]\]",
            rf"\[\[{re.escape(target)}\|[^\]]+\]\]",
            rf"-\s*\[\[{re.escape(target)}\]\].*\n?",
        ]

        removed = False
        for pattern in patterns:
            new_content, count = re.subn(pattern, "", source_content, flags=re.IGNORECASE)
            if count > 0:
                source_content = new_content
                removed = True

        if not removed:
            return {
                "success": False,
                "error": f"Link to {target} not found in {source}",
            }

        # Clean up empty Related sections
        source_content = re.sub(r"\n## Related Notes?\n\s*\n", "\n", source_content)

        # Write updated content
        source_path.write_text(source_content, encoding="utf-8")

        # Update database - remove the edge
        # Note: We'd need to add a delete_edge method to SQLiteDB
        logger.info(f"Removed link: {source_doc.id} -> {target}")

        return {
            "success": True,
            "action": "unlinked",
            "source": source_doc.id,
            "target": target,
        }

    def suggest_links(self, doc_id: str, top_k: int = 5) -> list[dict]:
        """Suggest potential links for a document.

        Args:
            doc_id: Document ID to suggest links for.
            top_k: Number of suggestions.

        Returns:
            List of suggested links with reasons.
        """
        suggestions = []

        # Get the document
        node = self.sqlite_db.get_node(doc_id)
        if not node:
            return suggestions

        # Get existing links
        outgoing = self.sqlite_db.get_outgoing_edges(doc_id)
        existing_targets = {e.target_id for e in outgoing}

        # Find documents with similar tags
        if node.tags:
            for tag in node.tags:
                all_nodes = self.sqlite_db.get_all_nodes()
                for other in all_nodes:
                    if other.id == doc_id or other.id in existing_targets:
                        continue
                    if other.tags and tag in other.tags:
                        suggestions.append(
                            {
                                "target_id": other.id,
                                "target_title": other.title,
                                "reason": f"Shares tag: {tag}",
                                "confidence": 0.7,
                            }
                        )

        # Find documents of same type
        same_type = self.sqlite_db.get_nodes_by_type(node.type)
        for other in same_type[:10]:
            if other.id == doc_id or other.id in existing_targets:
                continue
            if other.id not in [s["target_id"] for s in suggestions]:
                suggestions.append(
                    {
                        "target_id": other.id,
                        "target_title": other.title,
                        "reason": f"Same type: {node.type}",
                        "confidence": 0.5,
                    }
                )

        # Sort by confidence and limit
        suggestions.sort(key=lambda x: x["confidence"], reverse=True)
        return suggestions[:top_k]

    def find_orphans(self) -> list[dict]:
        """Find documents with no incoming or outgoing links.

        Returns:
            List of orphan documents.
        """
        all_nodes = self.sqlite_db.get_all_nodes()
        orphans = []

        for node in all_nodes:
            incoming = self.sqlite_db.get_incoming_edges(node.id)
            outgoing = self.sqlite_db.get_outgoing_edges(node.id)

            if not incoming and not outgoing:
                orphans.append(
                    {
                        "id": node.id,
                        "title": node.title,
                        "type": node.type,
                        "file_path": node.file_path,
                    }
                )

        return orphans

    def find_broken_links(self) -> list[dict]:
        """Find links pointing to non-existent documents.

        Returns:
            List of broken links.
        """
        broken = []

        for md_file in self.vault_path.rglob("*.md"):
            doc = self.parser.parse_file(md_file)

            for backlink in doc.backlinks:
                target_path = self.parser.resolve_link(backlink.target)
                if not target_path:
                    broken.append(
                        {
                            "source_id": doc.id,
                            "source_file": str(md_file),
                            "broken_link": backlink.target,
                            "context": backlink.context,
                        }
                    )

        return broken

    def _resolve_note(self, identifier: str) -> Path | None:
        """Resolve a note identifier to a file path.

        Args:
            identifier: Filename or document ID.

        Returns:
            Path to the file or None.
        """
        # Try as filename
        identifier_md = f"{identifier}.md" if not identifier.endswith(".md") else identifier

        direct_path = self.vault_path / identifier_md
        if direct_path.exists():
            return direct_path

        # Try by ID in database
        node = self.sqlite_db.get_node(identifier.replace(".md", ""))
        if node:
            path = Path(node.file_path)
            if path.exists():
                return path

        # Try resolve_link
        return self.parser.resolve_link(identifier.replace(".md", ""))
