"""File system watcher for real-time synchronization."""

import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from mini_docs_mcp.db.sqlite import Edge, Node, SQLiteDB
from mini_docs_mcp.db.vector import VectorDB, chunk_text
from mini_docs_mcp.parser.markdown import MarkdownParser

logger = logging.getLogger(__name__)


class VaultEventHandler(FileSystemEventHandler):
    """Handler for vault file system events."""

    def __init__(
        self,
        vault_path: Path,
        sqlite_db: SQLiteDB,
        vector_db: VectorDB,
        parser: MarkdownParser,
        debounce_seconds: float = 1.0,
    ):
        """Initialize the event handler.

        Args:
            vault_path: Path to the vault directory.
            sqlite_db: SQLite database instance.
            vector_db: Vector database instance.
            parser: Markdown parser instance.
            debounce_seconds: Time to wait before processing events.
        """
        self.vault_path = vault_path
        self.sqlite_db = sqlite_db
        self.vector_db = vector_db
        self.parser = parser
        self.debounce_seconds = debounce_seconds

        # Debounce tracking
        self._pending_events: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()
        self._processing_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start_processing(self) -> None:
        """Start the background processing thread."""
        self._stop_event.clear()
        self._processing_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._processing_thread.start()

    def stop_processing(self) -> None:
        """Stop the background processing thread."""
        self._stop_event.set()
        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)

    def _process_loop(self) -> None:
        """Background loop for processing debounced events."""
        while not self._stop_event.is_set():
            time.sleep(0.5)
            now = time.time()

            with self._lock:
                to_process = []
                for path, (event_type, timestamp) in list(self._pending_events.items()):
                    if now - timestamp >= self.debounce_seconds:
                        to_process.append((path, event_type))
                        del self._pending_events[path]

            for path, event_type in to_process:
                self._handle_event(path, event_type)

    def on_created(self, event: FileSystemEvent) -> None:
        """Handle file creation."""
        if not event.is_directory and event.src_path.endswith(".md"):
            self._queue_event(event.src_path, "created")

    def on_modified(self, event: FileSystemEvent) -> None:
        """Handle file modification."""
        if not event.is_directory and event.src_path.endswith(".md"):
            self._queue_event(event.src_path, "modified")

    def on_deleted(self, event: FileSystemEvent) -> None:
        """Handle file deletion."""
        if not event.is_directory and event.src_path.endswith(".md"):
            self._queue_event(event.src_path, "deleted")

    def on_moved(self, event: FileSystemEvent) -> None:
        """Handle file move/rename."""
        if not event.is_directory:
            if hasattr(event, "src_path") and event.src_path.endswith(".md"):
                self._queue_event(event.src_path, "deleted")
            if hasattr(event, "dest_path") and event.dest_path.endswith(".md"):
                self._queue_event(event.dest_path, "created")

    def _queue_event(self, path: str, event_type: str) -> None:
        """Queue an event for debounced processing."""
        # Ignore backup files
        if ".backups" in path:
            return

        with self._lock:
            self._pending_events[path] = (event_type, time.time())
        logger.debug(f"Queued event: {event_type} - {path}")

    def _handle_event(self, path: str, event_type: str) -> None:
        """Handle a file system event.

        Args:
            path: Path to the affected file.
            event_type: Type of event (created, modified, deleted).
        """
        file_path = Path(path)

        try:
            if event_type == "deleted":
                self._handle_deletion(file_path)
            else:
                self._handle_upsert(file_path)
        except Exception as e:
            logger.error(f"Error handling {event_type} event for {path}: {e}")

    def _handle_upsert(self, file_path: Path) -> None:
        """Handle file creation or modification."""
        if not file_path.exists():
            return

        # Parse the document
        doc = self.parser.parse_file(file_path)
        content = file_path.read_text(encoding="utf-8")

        # Check if content actually changed
        existing_node = self.sqlite_db.get_node(doc.id)
        new_hash = SQLiteDB.compute_hash(content)

        if existing_node and existing_node.hash == new_hash:
            logger.debug(f"No content change detected for: {doc.id}")
            return

        # Update SQLite
        node = Node(
            id=doc.id,
            file_path=str(file_path),
            title=doc.title,
            type=doc.type,
            status=doc.status,
            tags=doc.tags,
            hash=new_hash,
            summary=doc.get_summary(),
        )
        self.sqlite_db.upsert_node(node)

        # Update edges
        self.sqlite_db.delete_edges_for_source(doc.id)
        for backlink in doc.backlinks:
            target_path = self.parser.resolve_link(backlink.target)
            if target_path:
                target_doc = self.parser.parse_file(target_path)
                target_id = target_doc.id
            else:
                target_id = backlink.target.lower().replace(" ", "-")

            edge = Edge(
                id=None,
                source_id=doc.id,
                target_id=target_id,
                context_snippet=backlink.context,
            )
            self.sqlite_db.upsert_edge(edge)

        # Update vector DB
        chunks = chunk_text(doc.body)
        self.vector_db.update_document(doc.id, str(file_path), chunks)

        logger.info(f"Indexed: {doc.id}")

    def _handle_deletion(self, file_path: Path) -> None:
        """Handle file deletion."""
        # Find node by file path
        node = self.sqlite_db.get_node_by_path(str(file_path))
        if not node:
            # Try by stem
            doc_id = file_path.stem
            node = self.sqlite_db.get_node(doc_id)

        if node:
            self.sqlite_db.delete_node(node.id)
            self.vector_db.delete_document(node.id)
            logger.info(f"Removed from index: {node.id}")


class VaultWatcher:
    """File system watcher for the vault."""

    def __init__(
        self,
        vault_path: Path,
        sqlite_db: SQLiteDB,
        vector_db: VectorDB,
        parser: MarkdownParser,
    ):
        """Initialize the watcher.

        Args:
            vault_path: Path to the vault directory.
            sqlite_db: SQLite database instance.
            vector_db: Vector database instance.
            parser: Markdown parser instance.
        """
        self.vault_path = vault_path
        self.sqlite_db = sqlite_db
        self.vector_db = vector_db
        self.parser = parser

        self.handler = VaultEventHandler(vault_path, sqlite_db, vector_db, parser)
        self.observer = Observer()
        self._running = False

    def start(self) -> None:
        """Start watching the vault."""
        if self._running:
            return

        self.handler.start_processing()
        self.observer.schedule(self.handler, str(self.vault_path), recursive=True)
        self.observer.start()
        self._running = True
        logger.info(f"Started watching vault: {self.vault_path}")

    def stop(self) -> None:
        """Stop watching the vault."""
        if not self._running:
            return

        self.observer.stop()
        self.handler.stop_processing()
        self.observer.join(timeout=5.0)
        self._running = False
        logger.info("Stopped watching vault")

    @property
    def is_running(self) -> bool:
        """Check if the watcher is running."""
        return self._running


class VaultSyncer:
    """Full vault synchronization utility."""

    def __init__(
        self,
        vault_path: Path,
        sqlite_db: SQLiteDB,
        vector_db: VectorDB,
        parser: MarkdownParser,
    ):
        """Initialize the syncer.

        Args:
            vault_path: Path to the vault directory.
            sqlite_db: SQLite database instance.
            vector_db: Vector database instance.
            parser: Markdown parser instance.
        """
        self.vault_path = vault_path
        self.sqlite_db = sqlite_db
        self.vector_db = vector_db
        self.parser = parser

    def full_sync(
        self,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> dict:
        """Perform full synchronization of the vault.

        Args:
            progress_callback: Optional callback for progress updates.
                              Receives (current, total, filename).

        Returns:
            Sync statistics.
        """
        stats = {
            "files_scanned": 0,
            "files_indexed": 0,
            "files_updated": 0,
            "files_removed": 0,
            "errors": [],
        }

        # Get all markdown files
        md_files = list(self.vault_path.rglob("*.md"))
        # Exclude backup files
        md_files = [f for f in md_files if ".backups" not in str(f)]

        total = len(md_files)
        stats["files_scanned"] = total

        # Track existing files
        existing_paths = set()

        # Index each file
        for i, file_path in enumerate(md_files):
            if progress_callback:
                progress_callback(i + 1, total, file_path.name)

            try:
                doc = self.parser.parse_file(file_path)
                content = file_path.read_text(encoding="utf-8")
                new_hash = SQLiteDB.compute_hash(content)

                existing_paths.add(str(file_path))

                # Check if update needed
                existing_node = self.sqlite_db.get_node(doc.id)
                if existing_node and existing_node.hash == new_hash:
                    continue  # No changes

                # Update node
                node = Node(
                    id=doc.id,
                    file_path=str(file_path),
                    title=doc.title,
                    type=doc.type,
                    status=doc.status,
                    tags=doc.tags,
                    hash=new_hash,
                    summary=doc.get_summary(),
                )
                self.sqlite_db.upsert_node(node)

                # Update edges
                self.sqlite_db.delete_edges_for_source(doc.id)
                for backlink in doc.backlinks:
                    target_path = self.parser.resolve_link(backlink.target)
                    if target_path:
                        target_doc = self.parser.parse_file(target_path)
                        target_id = target_doc.id
                    else:
                        target_id = backlink.target.lower().replace(" ", "-")

                    edge = Edge(
                        id=None,
                        source_id=doc.id,
                        target_id=target_id,
                        context_snippet=backlink.context,
                    )
                    self.sqlite_db.upsert_edge(edge)

                # Update vector DB
                chunks = chunk_text(doc.body)
                self.vector_db.update_document(doc.id, str(file_path), chunks)

                if existing_node:
                    stats["files_updated"] += 1
                else:
                    stats["files_indexed"] += 1

            except Exception as e:
                stats["errors"].append({"file": str(file_path), "error": str(e)})
                logger.error(f"Error indexing {file_path}: {e}")

        # Remove nodes for deleted files
        all_nodes = self.sqlite_db.get_all_nodes()
        for node in all_nodes:
            if node.file_path not in existing_paths:
                self.sqlite_db.delete_node(node.id)
                self.vector_db.delete_document(node.id)
                stats["files_removed"] += 1

        return stats

    def check_consistency(self) -> dict:
        """Check consistency between files and database.

        Returns:
            Consistency report.
        """
        report = {
            "consistent": True,
            "missing_in_db": [],
            "missing_files": [],
            "hash_mismatches": [],
            "broken_links": [],
        }

        # Get all markdown files
        md_files = set(str(f) for f in self.vault_path.rglob("*.md") if ".backups" not in str(f))

        # Check each file against database
        for file_path in md_files:
            node = self.sqlite_db.get_node_by_path(file_path)
            if not node:
                report["missing_in_db"].append(file_path)
                report["consistent"] = False
                continue

            # Check hash
            content = Path(file_path).read_text(encoding="utf-8")
            current_hash = SQLiteDB.compute_hash(content)
            if node.hash != current_hash:
                report["hash_mismatches"].append(
                    {
                        "file": file_path,
                        "db_hash": node.hash,
                        "file_hash": current_hash,
                    }
                )
                report["consistent"] = False

        # Check for database entries without files
        all_nodes = self.sqlite_db.get_all_nodes()
        for node in all_nodes:
            if node.file_path not in md_files:
                report["missing_files"].append({"id": node.id, "expected_path": node.file_path})
                report["consistent"] = False

        # Check for broken links
        for node in all_nodes:
            outgoing = self.sqlite_db.get_outgoing_edges(node.id)
            for edge in outgoing:
                target_node = self.sqlite_db.get_node(edge.target_id)
                if not target_node:
                    report["broken_links"].append(
                        {
                            "source": node.id,
                            "target": edge.target_id,
                            "context": edge.context_snippet,
                        }
                    )

        return report
