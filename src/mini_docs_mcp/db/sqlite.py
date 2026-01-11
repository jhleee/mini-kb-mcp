"""SQLite database operations for nodes and edges."""

import hashlib
import json
import logging
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Node:
    """Represents a document node in the knowledge base."""

    id: str
    file_path: str
    title: str | None = None
    type: str = "note"
    status: str = "draft"
    tags: list[str] | None = None
    hash: str | None = None
    summary: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "file_path": self.file_path,
            "title": self.title,
            "type": self.type,
            "status": self.status,
            "tags": self.tags,
            "hash": self.hash,
            "summary": self.summary,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


@dataclass
class Edge:
    """Represents a backlink edge between two documents."""

    id: int | None
    source_id: str
    target_id: str
    context_snippet: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "context_snippet": self.context_snippet,
        }


class SQLiteDB:
    """SQLite database manager for the knowledge base."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS nodes (
        id TEXT PRIMARY KEY,
        file_path TEXT UNIQUE NOT NULL,
        title TEXT,
        type TEXT DEFAULT 'note',
        status TEXT DEFAULT 'draft',
        tags TEXT,
        hash TEXT,
        summary TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS edges (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id TEXT NOT NULL,
        target_id TEXT NOT NULL,
        context_snippet TEXT,
        FOREIGN KEY (source_id) REFERENCES nodes(id) ON DELETE CASCADE,
        UNIQUE(source_id, target_id)
    );

    CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id);
    CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id);
    CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type);
    CREATE INDEX IF NOT EXISTS idx_nodes_hash ON nodes(hash);
    """

    def __init__(self, db_path: Path | str):
        """Initialize the database.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.executescript(self.SCHEMA)
        logger.info(f"Database initialized at {self.db_path}")

    @staticmethod
    def compute_hash(content: str) -> str:
        """Compute MD5 hash of content."""
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    # Node operations
    def upsert_node(self, node: Node) -> None:
        """Insert or update a node."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO nodes (id, file_path, title, type, status, tags, hash, summary, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    file_path = excluded.file_path,
                    title = excluded.title,
                    type = excluded.type,
                    status = excluded.status,
                    tags = excluded.tags,
                    hash = excluded.hash,
                    summary = excluded.summary,
                    updated_at = excluded.updated_at
                """,
                (
                    node.id,
                    node.file_path,
                    node.title,
                    node.type,
                    node.status,
                    json.dumps(node.tags) if node.tags else None,
                    node.hash,
                    node.summary,
                    datetime.now().isoformat(),
                ),
            )
        logger.debug(f"Upserted node: {node.id}")

    def get_node(self, node_id: str) -> Node | None:
        """Get a node by ID."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM nodes WHERE id = ?", (node_id,)).fetchone()
            if row:
                return self._row_to_node(row)
        return None

    def get_node_by_path(self, file_path: str) -> Node | None:
        """Get a node by file path."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM nodes WHERE file_path = ?", (file_path,)).fetchone()
            if row:
                return self._row_to_node(row)
        return None

    def get_all_nodes(self) -> list[Node]:
        """Get all nodes."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM nodes ORDER BY updated_at DESC").fetchall()
            return [self._row_to_node(row) for row in rows]

    def get_nodes_by_type(self, type_: str) -> list[Node]:
        """Get nodes by type."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM nodes WHERE type = ? ORDER BY updated_at DESC", (type_,)
            ).fetchall()
            return [self._row_to_node(row) for row in rows]

    def get_recent_nodes(self, hours: int = 24) -> list[Node]:
        """Get recently modified nodes."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM nodes
                WHERE updated_at >= datetime('now', ?)
                ORDER BY updated_at DESC
                """,
                (f"-{hours} hours",),
            ).fetchall()
            return [self._row_to_node(row) for row in rows]

    def delete_node(self, node_id: str) -> bool:
        """Delete a node and its edges."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
            deleted = cursor.rowcount > 0
            if deleted:
                logger.debug(f"Deleted node: {node_id}")
            return deleted

    def _row_to_node(self, row: sqlite3.Row) -> Node:
        """Convert a database row to a Node object."""
        tags = json.loads(row["tags"]) if row["tags"] else None
        created_at = datetime.fromisoformat(row["created_at"]) if row["created_at"] else None
        updated_at = datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None
        return Node(
            id=row["id"],
            file_path=row["file_path"],
            title=row["title"],
            type=row["type"],
            status=row["status"],
            tags=tags,
            hash=row["hash"],
            summary=row["summary"],
            created_at=created_at,
            updated_at=updated_at,
        )

    # Edge operations
    def upsert_edge(self, edge: Edge) -> None:
        """Insert or update an edge."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO edges (source_id, target_id, context_snippet)
                VALUES (?, ?, ?)
                ON CONFLICT(source_id, target_id) DO UPDATE SET
                    context_snippet = excluded.context_snippet
                """,
                (edge.source_id, edge.target_id, edge.context_snippet),
            )
        logger.debug(f"Upserted edge: {edge.source_id} -> {edge.target_id}")

    def get_outgoing_edges(self, source_id: str) -> list[Edge]:
        """Get all edges originating from a node."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM edges WHERE source_id = ?", (source_id,)).fetchall()
            return [self._row_to_edge(row) for row in rows]

    def get_incoming_edges(self, target_id: str) -> list[Edge]:
        """Get all edges pointing to a node."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM edges WHERE target_id = ?", (target_id,)).fetchall()
            return [self._row_to_edge(row) for row in rows]

    def delete_edges_for_source(self, source_id: str) -> int:
        """Delete all edges originating from a node."""
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM edges WHERE source_id = ?", (source_id,))
            return cursor.rowcount

    def _row_to_edge(self, row: sqlite3.Row) -> Edge:
        """Convert a database row to an Edge object."""
        return Edge(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            context_snippet=row["context_snippet"],
        )

    # Statistics
    def get_stats(self) -> dict:
        """Get database statistics."""
        with self._get_connection() as conn:
            total_nodes = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]
            total_edges = conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0]

            # Orphan nodes (no incoming or outgoing edges)
            orphans = conn.execute(
                """
                SELECT COUNT(*) FROM nodes n
                WHERE NOT EXISTS (SELECT 1 FROM edges WHERE source_id = n.id)
                AND NOT EXISTS (SELECT 1 FROM edges WHERE target_id = n.id)
                """
            ).fetchone()[0]

            # Most referenced nodes
            top_referenced = conn.execute(
                """
                SELECT n.id, n.title, COUNT(e.id) as ref_count
                FROM nodes n
                LEFT JOIN edges e ON n.id = e.target_id
                GROUP BY n.id
                ORDER BY ref_count DESC
                LIMIT 10
                """
            ).fetchall()

            # Type distribution
            type_dist = conn.execute(
                """
                SELECT type, COUNT(*) as count
                FROM nodes
                GROUP BY type
                """
            ).fetchall()

            return {
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "orphan_nodes": orphans,
                "top_referenced": [
                    {"id": r["id"], "title": r["title"], "ref_count": r["ref_count"]}
                    for r in top_referenced
                ],
                "type_distribution": {r["type"]: r["count"] for r in type_dist},
            }

    def get_all_tags(self) -> dict[str, int]:
        """Get all tags with document counts."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT tags FROM nodes WHERE tags IS NOT NULL").fetchall()

        tag_counts: dict[str, int] = {}
        for row in rows:
            tags = json.loads(row["tags"])
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        return dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True))
