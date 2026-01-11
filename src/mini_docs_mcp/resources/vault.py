"""MCP Resources for vault information."""

import logging
from pathlib import Path

from ..db.sqlite import SQLiteDB
from ..db.vector import VectorDB

logger = logging.getLogger(__name__)


class VaultResources:
    """Provider for vault-related MCP resources."""

    def __init__(
        self,
        vault_path: Path,
        sqlite_db: SQLiteDB,
        vector_db: VectorDB,
    ):
        """Initialize the resource provider.

        Args:
            vault_path: Path to the vault directory.
            sqlite_db: SQLite database instance.
            vector_db: Vector database instance.
        """
        self.vault_path = vault_path
        self.sqlite_db = sqlite_db
        self.vector_db = vector_db

    def get_graph_summary(self) -> str:
        """Get knowledge base health report.

        Returns:
            Formatted summary of the knowledge base.
        """
        sqlite_stats = self.sqlite_db.get_stats()
        vector_stats = self.vector_db.get_stats()

        # Build summary
        lines = [
            "# Knowledge Base Summary",
            "",
            "## Overview",
            f"- **Total Documents**: {sqlite_stats['total_nodes']}",
            f"- **Total Links**: {sqlite_stats['total_edges']}",
            f"- **Orphan Documents**: {sqlite_stats['orphan_nodes']}",
            f"- **Vector Chunks**: {vector_stats['total_chunks']}",
            "",
            "## Document Types",
        ]

        for doc_type, count in sqlite_stats.get("type_distribution", {}).items():
            lines.append(f"- {doc_type}: {count}")

        lines.extend(
            [
                "",
                "## Most Referenced Documents",
            ]
        )

        for ref in sqlite_stats.get("top_referenced", [])[:5]:
            if ref["ref_count"] > 0:
                lines.append(f"- **{ref['title'] or ref['id']}**: {ref['ref_count']} references")

        # Health indicators
        lines.extend(
            [
                "",
                "## Health Indicators",
            ]
        )

        total_nodes = sqlite_stats["total_nodes"]
        if total_nodes > 0:
            orphan_ratio = sqlite_stats["orphan_nodes"] / total_nodes
            if orphan_ratio > 0.5:
                lines.append("- ⚠️ High orphan ratio: Many documents are not linked")
            elif orphan_ratio > 0.2:
                lines.append("- 🔶 Moderate orphan ratio: Consider adding more links")
            else:
                lines.append("- ✅ Good link coverage")

            avg_refs = sqlite_stats["total_edges"] / total_nodes
            if avg_refs < 1:
                lines.append("- ⚠️ Low average references per document")
            else:
                lines.append(f"- ✅ Average {avg_refs:.1f} links per document")
        else:
            lines.append("- 📭 Vault is empty")

        return "\n".join(lines)

    def get_recent_notes(self, hours: int = 24) -> str:
        """Get recently modified notes.

        Args:
            hours: Look back period in hours.

        Returns:
            Formatted list of recent notes.
        """
        recent_nodes = self.sqlite_db.get_recent_nodes(hours=hours)

        if not recent_nodes:
            return f"No documents modified in the last {hours} hours."

        lines = [
            f"# Recently Modified Documents (Last {hours} hours)",
            "",
        ]

        for node in recent_nodes:
            updated = node.updated_at.strftime("%Y-%m-%d %H:%M") if node.updated_at else "Unknown"
            lines.append(f"- **{node.title or node.id}** ({node.type}) - Updated: {updated}")
            if node.summary:
                lines.append(f"  > {node.summary[:100]}...")

        return "\n".join(lines)

    def get_all_tags(self) -> str:
        """Get all tags with document counts.

        Returns:
            Formatted list of tags.
        """
        tag_counts = self.sqlite_db.get_all_tags()

        if not tag_counts:
            return "No tags found in the vault."

        lines = [
            "# All Tags",
            "",
            "| Tag | Documents |",
            "|-----|-----------|",
        ]

        for tag, count in tag_counts.items():
            lines.append(f"| {tag} | {count} |")

        return "\n".join(lines)

    def get_orphan_documents(self) -> str:
        """Get documents with no links.

        Returns:
            Formatted list of orphan documents.
        """
        all_nodes = self.sqlite_db.get_all_nodes()
        orphans = []

        for node in all_nodes:
            incoming = self.sqlite_db.get_incoming_edges(node.id)
            outgoing = self.sqlite_db.get_outgoing_edges(node.id)

            if not incoming and not outgoing:
                orphans.append(node)

        if not orphans:
            return "No orphan documents found. All documents are well-connected!"

        lines = [
            "# Orphan Documents",
            "",
            "These documents have no incoming or outgoing links:",
            "",
        ]

        for node in orphans:
            lines.append(f"- **{node.title or node.id}** ({node.type})")
            if node.summary:
                lines.append(f"  > {node.summary[:100]}...")

        lines.extend(
            [
                "",
                "Consider adding links to connect these documents to the knowledge graph.",
            ]
        )

        return "\n".join(lines)

    def get_document_type_overview(self, doc_type: str) -> str:
        """Get overview of documents by type.

        Args:
            doc_type: Document type to filter.

        Returns:
            Formatted list of documents.
        """
        nodes = self.sqlite_db.get_nodes_by_type(doc_type)

        if not nodes:
            return f"No documents found with type: {doc_type}"

        lines = [
            f"# Documents of Type: {doc_type}",
            "",
            f"Total: {len(nodes)} documents",
            "",
        ]

        for node in nodes[:20]:  # Limit to 20
            lines.append(f"## {node.title or node.id}")
            if node.tags:
                lines.append(f"Tags: {', '.join(node.tags)}")
            if node.summary:
                lines.append(f"> {node.summary}")
            lines.append("")

        if len(nodes) > 20:
            lines.append(f"... and {len(nodes) - 20} more documents")

        return "\n".join(lines)
