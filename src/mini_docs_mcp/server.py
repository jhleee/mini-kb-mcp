"""FastMCP Server for Active Markdown Vault."""

import logging
import os
from pathlib import Path

from fastmcp import FastMCP

from mini_docs_mcp.db.sqlite import SQLiteDB
from mini_docs_mcp.db.vector import VectorDB, create_default_embedding_function
from mini_docs_mcp.parser.markdown import MarkdownParser
from mini_docs_mcp.resources.vault import VaultResources
from mini_docs_mcp.tools.link import LinkEngine
from mini_docs_mcp.tools.read import ReadEngine
from mini_docs_mcp.tools.search import SearchEngine
from mini_docs_mcp.tools.write import WriteEngine
from mini_docs_mcp.watcher.sync import VaultSyncer, VaultWatcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("mini-docs-mcp")

# Global instances (initialized on startup)
_vault_path: Path | None = None
_sqlite_db: SQLiteDB | None = None
_vector_db: VectorDB | None = None
_parser: MarkdownParser | None = None
_search_engine: SearchEngine | None = None
_read_engine: ReadEngine | None = None
_write_engine: WriteEngine | None = None
_link_engine: LinkEngine | None = None
_resources: VaultResources | None = None
_watcher: VaultWatcher | None = None
_syncer: VaultSyncer | None = None


def _get_vault_path() -> Path:
    """Get the vault path from environment or default."""
    env_path = os.environ.get("VAULT_PATH")
    if env_path:
        return Path(env_path)
    # Default: vault directory relative to this file's parent
    return Path(__file__).parent.parent.parent / "vault"


def _ensure_initialized():
    """Ensure all components are initialized."""
    global _vault_path, _sqlite_db, _vector_db, _parser
    global _search_engine, _read_engine, _write_engine, _link_engine
    global _resources, _watcher, _syncer

    if _sqlite_db is not None:
        return

    _vault_path = _get_vault_path()
    _vault_path.mkdir(parents=True, exist_ok=True)

    # Data directories
    data_dir = _vault_path.parent / ".mini_docs_data"
    data_dir.mkdir(exist_ok=True)

    # Initialize databases
    _sqlite_db = SQLiteDB(data_dir / "metadata.db")
    _vector_db = VectorDB(data_dir / "vectors")

    # Set up embedding function
    logger.info("Loading embedding model...")
    embedding_fn = create_default_embedding_function()
    _vector_db.set_embedding_function(embedding_fn)
    logger.info("Embedding model loaded")

    # Initialize parser and engines
    _parser = MarkdownParser(_vault_path)
    _search_engine = SearchEngine(_sqlite_db, _vector_db)
    _read_engine = ReadEngine(_vault_path, _sqlite_db, _parser)
    _write_engine = WriteEngine(_vault_path, _sqlite_db, _vector_db, _parser)
    _link_engine = LinkEngine(_vault_path, _sqlite_db, _parser, _write_engine)
    _resources = VaultResources(_vault_path, _sqlite_db, _vector_db)
    _syncer = VaultSyncer(_vault_path, _sqlite_db, _vector_db, _parser)
    _watcher = VaultWatcher(_vault_path, _sqlite_db, _vector_db, _parser)

    # Initial sync
    logger.info(f"Performing initial sync of vault: {_vault_path}")
    stats = _syncer.full_sync()
    logger.info(
        f"Initial sync complete: {stats['files_indexed']} indexed, "
        f"{stats['files_updated']} updated, {len(stats['errors'])} errors"
    )

    # Start watcher
    _watcher.start()
    logger.info("File watcher started")


# ============================================================================
# MCP Tools
# ============================================================================


@mcp.tool()
async def search_vault(
    query: str,
    type: str | None = None,
    tags: list[str] | None = None,
    top_k: int = 10,
    expand_graph: bool = False,
) -> dict:
    """Search the vault using hybrid semantic + metadata search.

    Performs intelligent search combining vector similarity with metadata
    filtering and backlink graph weighting.

    Args:
        query: Natural language search query
        type: Filter by document type (concept, task, ref, log, idea)
        tags: Filter by tags (documents must have at least one of these tags)
        top_k: Maximum number of results to return (default: 10)
        expand_graph: If True, include 1-hop connected documents in results

    Returns:
        Search results with relevance scores, metadata, and optional graph context
    """
    _ensure_initialized()

    if expand_graph:
        results = _search_engine.search_with_expansion(
            query=query,
            top_k=top_k,
            expansion_hops=1,
        )
    else:
        results = _search_engine.search(
            query=query,
            type_filter=type,
            tag_filter=tags,
            top_k=top_k,
            include_graph_context=True,
        )

    return results


@mcp.tool()
async def read_note(
    filename: str,
    include_related: bool = True,
) -> dict:
    """Read a note from the vault with its full content and relationships.

    Retrieves the complete document including metadata, content, and
    information about linked documents.

    Args:
        filename: Note filename (with or without .md extension) or document ID
        include_related: Whether to include summaries of related documents

    Returns:
        Note content, metadata, backlinks, and related document summaries
    """
    _ensure_initialized()

    return _read_engine.read_note(
        filename=filename,
        include_backlinks=True,
        include_related=include_related,
    )


@mcp.tool()
async def write_note(
    filename: str,
    content: str,
    title: str | None = None,
    type: str = "note",
    tags: list[str] | None = None,
    status: str = "draft",
) -> dict:
    """Create or overwrite a note in the vault.

    Creates a new document or completely replaces an existing one.
    The document is automatically indexed for search.

    Args:
        filename: Note filename (with or without .md extension)
        content: The body content of the note (markdown format)
        title: Document title (defaults to filename if not provided)
        type: Document type (concept, task, ref, log, idea, note)
        tags: List of tags for categorization
        status: Document status (draft, verified, superseded)

    Returns:
        Operation result with file path and action taken
    """
    _ensure_initialized()

    metadata = {
        "type": type,
        "status": status,
    }
    if title:
        metadata["title"] = title
    if tags:
        metadata["tags"] = tags

    return _write_engine.write_note(
        filename=filename,
        content=content,
        metadata=metadata,
        auto_index=True,
    )


@mcp.tool()
async def patch_note(
    filename: str,
    section: str,
    content: str,
) -> dict:
    """Update a specific section of an existing note.

    Performs precise editing by replacing only the content under a
    specific header, preserving the rest of the document.

    Args:
        filename: Note filename or document ID
        section: Section header to update (e.g., "## Overview", "### Details")
        content: New content for the section (without the header)

    Returns:
        Operation result or available sections if not found
    """
    _ensure_initialized()

    return _write_engine.patch_note(
        filename=filename,
        section=section,
        new_content=content,
        auto_index=True,
    )


@mcp.tool()
async def link_notes(
    source: str,
    target: str,
    reason: str | None = None,
    bidirectional: bool = False,
) -> dict:
    """Create a backlink between two notes.

    Adds a [[backlink]] from the source document to the target document,
    typically in a "Related Notes" section.

    Args:
        source: Source note filename or ID (where the link will be added)
        target: Target note filename or ID (what the link points to)
        reason: Optional context explaining why these notes are related
        bidirectional: If True, also add a link from target to source

    Returns:
        Operation result including link status
    """
    _ensure_initialized()

    return _link_engine.link_notes(
        source=source,
        target=target,
        reason=reason,
        bidirectional=bidirectional,
    )


@mcp.tool()
async def sync_status() -> dict:
    """Check vault synchronization status and optionally force re-sync.

    Verifies consistency between the file system and database indexes.
    Reports any discrepancies and can trigger a full re-synchronization.

    Returns:
        Sync status report including any inconsistencies found
    """
    _ensure_initialized()

    # Check consistency
    report = _syncer.check_consistency()

    if not report["consistent"]:
        # Perform full sync if inconsistent
        logger.info("Inconsistencies found, performing full sync")
        sync_stats = _syncer.full_sync()
        report["sync_performed"] = True
        report["sync_stats"] = sync_stats
    else:
        report["sync_performed"] = False

    return report


@mcp.tool()
async def list_notes(
    type: str | None = None,
    tag: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """List notes in the vault with optional filtering.

    Args:
        type: Filter by document type
        tag: Filter by tag
        limit: Maximum number of results

    Returns:
        List of note summaries
    """
    _ensure_initialized()

    return _read_engine.list_notes(
        type_filter=type,
        tag_filter=tag,
        limit=limit,
    )


@mcp.tool()
async def find_similar(doc_id: str, top_k: int = 5) -> list[dict]:
    """Find documents similar to a given document.

    Uses semantic similarity to find related documents.

    Args:
        doc_id: Document ID to find similar documents for
        top_k: Number of results

    Returns:
        List of similar documents with similarity scores
    """
    _ensure_initialized()

    return _search_engine.find_similar(doc_id, top_k=top_k)


@mcp.tool()
async def suggest_links(doc_id: str, top_k: int = 5) -> list[dict]:
    """Suggest potential links for a document.

    Analyzes the document and suggests other documents that might be
    worth linking to based on shared tags, types, and content similarity.

    Args:
        doc_id: Document ID to suggest links for
        top_k: Number of suggestions

    Returns:
        List of suggested links with reasons
    """
    _ensure_initialized()

    return _link_engine.suggest_links(doc_id, top_k=top_k)


@mcp.tool()
async def find_broken_links() -> list[dict]:
    """Find all broken links in the vault.

    Scans all documents for backlinks pointing to non-existent documents.

    Returns:
        List of broken links with source and context
    """
    _ensure_initialized()

    return _link_engine.find_broken_links()


@mcp.tool()
async def get_note_graph(doc_id: str, depth: int = 1) -> dict:
    """Get the graph context around a note.

    Returns nodes and edges within the specified depth from the given document.

    Args:
        doc_id: Central document ID
        depth: How many hops to include (1 or 2)

    Returns:
        Graph structure with nodes and edges
    """
    _ensure_initialized()

    return _read_engine.get_note_graph(doc_id, depth=min(depth, 2))


# ============================================================================
# MCP Resources
# ============================================================================


@mcp.resource("vault://graph/summary")
async def graph_summary() -> str:
    """Get knowledge base health report.

    Provides statistics about the vault including document counts,
    type distribution, most referenced documents, and health indicators.
    """
    _ensure_initialized()
    return _resources.get_graph_summary()


@mcp.resource("vault://note/recent")
async def recent_notes() -> str:
    """Get recently modified notes (last 24 hours).

    Lists documents that were created or updated within the last day.
    """
    _ensure_initialized()
    return _resources.get_recent_notes(hours=24)


@mcp.resource("vault://tags/all")
async def all_tags() -> str:
    """Get all tags with document counts.

    Provides a complete list of tags used in the vault and how many
    documents use each tag.
    """
    _ensure_initialized()
    return _resources.get_all_tags()


@mcp.resource("vault://orphans")
async def orphan_documents() -> str:
    """Get documents with no links.

    Lists documents that have no incoming or outgoing backlinks,
    which may indicate they need to be better integrated into the
    knowledge graph.
    """
    _ensure_initialized()
    return _resources.get_orphan_documents()


def main():
    """Entry point for the MCP server."""
    logger.info("Starting mini-docs-mcp server...")
    _ensure_initialized()
    logger.info(f"Vault path: {_vault_path}")

    # Check for SSE transport mode
    transport = os.environ.get("MCP_TRANSPORT", "stdio").lower()

    if transport == "sse":
        host = os.environ.get("MCP_HOST", "0.0.0.0")
        port = int(os.environ.get("MCP_PORT", "8000"))
        logger.info(f"Starting SSE server on {host}:{port}")
        mcp.run(transport="sse", host=host, port=port)
    else:
        logger.info("Starting stdio server")
        mcp.run()


if __name__ == "__main__":
    main()
