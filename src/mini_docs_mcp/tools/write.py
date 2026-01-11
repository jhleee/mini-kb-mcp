"""Write tools for creating and modifying vault documents."""

import logging
import shutil
from datetime import datetime
from pathlib import Path

import frontmatter

from mini_docs_mcp.db.sqlite import Node, SQLiteDB
from mini_docs_mcp.db.vector import VectorDB, chunk_text
from mini_docs_mcp.parser.markdown import MarkdownParser

logger = logging.getLogger(__name__)


class WriteEngine:
    """Engine for writing and modifying vault documents."""

    def __init__(
        self,
        vault_path: Path,
        sqlite_db: SQLiteDB,
        vector_db: VectorDB,
        parser: MarkdownParser,
        backup_enabled: bool = True,
    ):
        """Initialize the write engine.

        Args:
            vault_path: Path to the vault directory.
            sqlite_db: SQLite database instance.
            vector_db: Vector database instance.
            parser: Markdown parser instance.
            backup_enabled: Whether to create backups before modifications.
        """
        self.vault_path = vault_path
        self.sqlite_db = sqlite_db
        self.vector_db = vector_db
        self.parser = parser
        self.backup_enabled = backup_enabled
        self.backup_dir = vault_path / ".backups"

        if backup_enabled:
            self.backup_dir.mkdir(exist_ok=True)

    def write_note(
        self,
        filename: str,
        content: str,
        metadata: dict | None = None,
        auto_index: bool = True,
    ) -> dict:
        """Create or overwrite a note.

        Args:
            filename: Note filename (with or without .md extension).
            content: Full content or body content.
            metadata: Optional metadata to set/override.
            auto_index: Whether to automatically index the note.

        Returns:
            Operation result.
        """
        # Normalize filename
        if not filename.endswith(".md"):
            filename = f"{filename}.md"

        file_path = self.vault_path / filename

        # Check if file exists
        is_new = not file_path.exists()

        # If file exists, check for conflicts
        if not is_new:
            existing_node = self.sqlite_db.get_node_by_path(str(file_path))
            if existing_node:
                current_content = file_path.read_text(encoding="utf-8")
                current_hash = SQLiteDB.compute_hash(current_content)
                if existing_node.hash and existing_node.hash != current_hash:
                    return {
                        "success": False,
                        "error": "Conflict detected: file was modified externally",
                        "suggestion": "Use sync_status to update the database first",
                    }

            # Create backup
            if self.backup_enabled:
                self._create_backup(file_path)

        # Prepare content with frontmatter
        final_content = self._prepare_content(content, metadata, filename)

        # Write file
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(final_content, encoding="utf-8")
            logger.info(f"{'Created' if is_new else 'Updated'} note: {filename}")
        except Exception as e:
            logger.error(f"Error writing {filename}: {e}")
            return {
                "success": False,
                "error": f"Failed to write file: {str(e)}",
            }

        # Index the note
        if auto_index:
            self._index_note(file_path)

        return {
            "success": True,
            "action": "created" if is_new else "updated",
            "file_path": str(file_path),
            "filename": filename,
        }

    def patch_note(
        self,
        filename: str,
        section: str,
        new_content: str,
        auto_index: bool = True,
    ) -> dict:
        """Update a specific section of a note.

        Args:
            filename: Note filename.
            section: Section header to update (e.g., "## Overview").
            new_content: New content for the section (without header).
            auto_index: Whether to automatically re-index.

        Returns:
            Operation result.
        """
        # Normalize filename
        if not filename.endswith(".md"):
            filename = f"{filename}.md"

        file_path = self.vault_path / filename
        if not file_path.exists():
            # Try to find by ID
            node = self.sqlite_db.get_node(Path(filename).stem)
            if node:
                file_path = Path(node.file_path)
            else:
                return {
                    "success": False,
                    "error": f"Note not found: {filename}",
                }

        # Read current content
        current_content = file_path.read_text(encoding="utf-8")

        # Check for conflicts
        existing_node = self.sqlite_db.get_node_by_path(str(file_path))
        if existing_node:
            current_hash = SQLiteDB.compute_hash(current_content)
            if existing_node.hash and existing_node.hash != current_hash:
                return {
                    "success": False,
                    "error": "Conflict detected: file was modified externally",
                }

        # Create backup
        if self.backup_enabled:
            self._create_backup(file_path)

        # Replace section
        updated_content = self.parser.replace_section(current_content, section, new_content)
        if updated_content is None:
            return {
                "success": False,
                "error": f"Section not found: {section}",
                "available_sections": [
                    s["header"] for s in self.parser.get_sections(current_content)
                ],
            }

        # Update timestamp in frontmatter
        updated_content = self.parser.update_frontmatter(
            updated_content, {"updated": datetime.now().isoformat()}
        )

        # Write file
        try:
            file_path.write_text(updated_content, encoding="utf-8")
            logger.info(f"Patched section '{section}' in: {filename}")
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to write file: {str(e)}",
            }

        # Re-index
        if auto_index:
            self._index_note(file_path)

        return {
            "success": True,
            "action": "patched",
            "section": section,
            "file_path": str(file_path),
        }

    def delete_note(self, filename: str, soft_delete: bool = True) -> dict:
        """Delete a note.

        Args:
            filename: Note filename.
            soft_delete: If True, move to backups instead of deleting.

        Returns:
            Operation result.
        """
        # Normalize filename
        if not filename.endswith(".md"):
            filename = f"{filename}.md"

        file_path = self.vault_path / filename
        if not file_path.exists():
            node = self.sqlite_db.get_node(Path(filename).stem)
            if node:
                file_path = Path(node.file_path)
            else:
                return {
                    "success": False,
                    "error": f"Note not found: {filename}",
                }

        # Get document ID
        try:
            doc = self.parser.parse_file(file_path)
            doc_id = doc.id
        except Exception:
            doc_id = file_path.stem

        # Soft delete or permanent delete
        if soft_delete and self.backup_enabled:
            backup_path = self._create_backup(file_path, prefix="deleted_")
            file_path.unlink()
            action = f"moved to {backup_path}"
        else:
            file_path.unlink()
            action = "permanently deleted"

        # Remove from databases
        self.sqlite_db.delete_node(doc_id)
        self.vector_db.delete_document(doc_id)

        logger.info(f"Deleted note: {filename} ({action})")

        return {
            "success": True,
            "action": action,
            "deleted_id": doc_id,
        }

    def _prepare_content(self, content: str, metadata: dict | None, filename: str) -> str:
        """Prepare content with proper frontmatter.

        Args:
            content: Raw content (may or may not have frontmatter).
            metadata: Additional metadata to set.
            filename: Filename for default ID.

        Returns:
            Content with frontmatter.
        """
        # Check if content already has frontmatter
        if content.strip().startswith("---"):
            post = frontmatter.loads(content)
        else:
            post = frontmatter.Post(content)

        # Set defaults
        defaults = {
            "id": Path(filename).stem,
            "title": Path(filename).stem.replace("-", " ").title(),
            "type": "note",
            "status": "draft",
            "updated": datetime.now().isoformat(),
        }

        for key, value in defaults.items():
            if key not in post.metadata:
                post.metadata[key] = value

        # Override with provided metadata
        if metadata:
            post.metadata.update(metadata)

        return frontmatter.dumps(post)

    def _create_backup(self, file_path: Path, prefix: str = "") -> Path:
        """Create a backup of a file.

        Args:
            file_path: Path to the file.
            prefix: Prefix for backup filename.

        Returns:
            Path to the backup file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{prefix}{file_path.stem}_{timestamp}{file_path.suffix}"
        backup_path = self.backup_dir / backup_name
        shutil.copy2(file_path, backup_path)
        logger.debug(f"Created backup: {backup_path}")
        return backup_path

    def _index_note(self, file_path: Path) -> None:
        """Index a note in the databases.

        Args:
            file_path: Path to the note.
        """
        try:
            doc = self.parser.parse_file(file_path)
            content = file_path.read_text(encoding="utf-8")

            # Update SQLite
            node = Node(
                id=doc.id,
                file_path=str(file_path),
                title=doc.title,
                type=doc.type,
                status=doc.status,
                tags=doc.tags,
                hash=SQLiteDB.compute_hash(content),
                summary=doc.get_summary(),
            )
            self.sqlite_db.upsert_node(node)

            # Update edges (backlinks)
            self.sqlite_db.delete_edges_for_source(doc.id)
            for backlink in doc.backlinks:
                # Try to resolve the target
                target_path = self.parser.resolve_link(backlink.target)
                if target_path:
                    target_doc = self.parser.parse_file(target_path)
                    target_id = target_doc.id
                else:
                    target_id = backlink.target.lower().replace(" ", "-")

                from ..db.sqlite import Edge

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

            logger.debug(f"Indexed note: {doc.id}")

        except Exception as e:
            logger.error(f"Error indexing {file_path}: {e}")
