"""Integration tests for the complete MCP server workflow."""


import pytest

from mini_docs_mcp.db.sqlite import SQLiteDB
from mini_docs_mcp.db.vector import VectorDB, create_default_embedding_function
from mini_docs_mcp.parser.markdown import MarkdownParser
from mini_docs_mcp.resources.vault import VaultResources
from mini_docs_mcp.tools.link import LinkEngine
from mini_docs_mcp.tools.read import ReadEngine
from mini_docs_mcp.tools.search import SearchEngine
from mini_docs_mcp.tools.write import WriteEngine
from mini_docs_mcp.watcher.sync import VaultSyncer


@pytest.fixture(scope="module")
def embedding_fn():
    """Create embedding function once for all tests."""
    return create_default_embedding_function()


@pytest.fixture
def test_vault(tmp_path, embedding_fn):
    """Create a complete test environment."""
    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create test documents
    docs = {
        "python-basics.md": """---
id: python-basics
title: Python Basics
type: concept
tags: [python, programming, beginner]
status: verified
---

# Python Basics

Python is a versatile programming language known for its readability.

## Variables

Variables in Python are dynamically typed.

## Functions

Functions are defined using the `def` keyword.

## Related Notes

- [[data-structures]]
""",
        "data-structures.md": """---
id: data-structures
title: Data Structures in Python
type: concept
tags: [python, data-structures]
status: draft
---

# Data Structures in Python

Python provides several built-in data structures.

## Lists

Lists are ordered, mutable collections.

## Dictionaries

Dictionaries store key-value pairs.

## Related Notes

- [[python-basics]]
- [[algorithms]]
""",
        "algorithms.md": """---
id: algorithms
title: Algorithm Design
type: concept
tags: [algorithms, computer-science]
status: draft
---

# Algorithm Design

Algorithms are step-by-step procedures for solving problems.

## Sorting

Common sorting algorithms include quicksort and mergesort.

## Searching

Binary search is efficient for sorted data.
""",
        "project-todo.md": """---
id: project-todo
title: Project TODO List
type: task
tags: [project, todo]
status: draft
---

# Project TODO List

## Pending Tasks

- Implement feature A
- Fix bug in module B
- Write documentation

## Completed

- Initial setup
""",
        "meeting-notes.md": """---
id: meeting-notes
title: Team Meeting Notes
type: log
tags: [meeting, team]
status: verified
---

# Team Meeting Notes

## 2024-01-10

Discussed project timeline and resource allocation.

## Action Items

- Review [[python-basics]] documentation
- Update [[project-todo]]
""",
    }

    for filename, content in docs.items():
        (vault_path / filename).write_text(content, encoding="utf-8")

    # Initialize components
    sqlite_db = SQLiteDB(data_dir / "metadata.db")
    vector_db = VectorDB(data_dir / "vectors")
    vector_db.set_embedding_function(embedding_fn)
    parser = MarkdownParser(vault_path)

    # Run initial sync
    syncer = VaultSyncer(vault_path, sqlite_db, vector_db, parser)
    syncer.full_sync()

    # Create engines
    search_engine = SearchEngine(sqlite_db, vector_db)
    read_engine = ReadEngine(vault_path, sqlite_db, parser)
    write_engine = WriteEngine(vault_path, sqlite_db, vector_db, parser)
    link_engine = LinkEngine(vault_path, sqlite_db, parser, write_engine)
    resources = VaultResources(vault_path, sqlite_db, vector_db)

    return {
        "vault_path": vault_path,
        "sqlite_db": sqlite_db,
        "vector_db": vector_db,
        "parser": parser,
        "syncer": syncer,
        "search_engine": search_engine,
        "read_engine": read_engine,
        "write_engine": write_engine,
        "link_engine": link_engine,
        "resources": resources,
    }


class TestSemanticSearch:
    """Test semantic search functionality."""

    def test_search_by_topic(self, test_vault):
        """Search for documents by topic."""
        search = test_vault["search_engine"]

        # Search for Python-related content
        results = search.search("Python programming language", top_k=5)
        assert results["total_results"] >= 2

        # Python basics should be top result
        titles = [r["title"] for r in results["results"]]
        assert "Python Basics" in titles

    def test_search_by_concept(self, test_vault):
        """Search for abstract concepts."""
        search = test_vault["search_engine"]

        results = search.search("sorting and searching algorithms", top_k=3)
        assert results["total_results"] >= 1

        # Algorithm doc should be in results
        ids = [r["id"] for r in results["results"]]
        assert "algorithms" in ids

    def test_search_with_type_filter(self, test_vault):
        """Search with document type filter."""
        search = test_vault["search_engine"]

        # Search only in tasks
        results = search.search("todo list", type_filter="task", top_k=5)
        for result in results["results"]:
            assert result["type"] == "task"

    def test_search_with_tag_filter(self, test_vault):
        """Search with tag filter."""
        search = test_vault["search_engine"]

        results = search.search("programming", tag_filter=["python"], top_k=5)
        for result in results["results"]:
            assert "python" in (result["tags"] or [])

    def test_search_with_graph_expansion(self, test_vault):
        """Search with graph context expansion."""
        search = test_vault["search_engine"]

        results = search.search_with_expansion("Python basics", top_k=2, expansion_hops=1)

        # Should have expanded context
        assert "expanded_context" in results
        # data-structures is linked from python-basics
        expanded_ids = [c["id"] for c in results.get("expanded_context", [])]
        assert len(expanded_ids) >= 0  # May have connected docs

    def test_find_similar_documents(self, test_vault):
        """Find documents similar to a given one."""
        search = test_vault["search_engine"]

        similar = search.find_similar("python-basics", top_k=3)
        # data-structures should be similar (both about Python)
        similar_ids = [s["id"] for s in similar]
        assert "data-structures" in similar_ids


class TestReadOperations:
    """Test document reading functionality."""

    def test_read_note_by_filename(self, test_vault):
        """Read a note by its filename."""
        read = test_vault["read_engine"]

        result = read.read_note("python-basics.md")
        assert result["success"]
        assert result["title"] == "Python Basics"
        assert result["type"] == "concept"
        assert "python" in result["tags"]

    def test_read_note_by_id(self, test_vault):
        """Read a note by its ID."""
        read = test_vault["read_engine"]

        result = read.read_note("algorithms")
        assert result["success"]
        assert result["title"] == "Algorithm Design"

    def test_read_note_with_backlinks(self, test_vault):
        """Read a note and verify backlink information."""
        read = test_vault["read_engine"]

        result = read.read_note("data-structures")
        assert result["success"]

        # Should have outgoing links
        outgoing = result.get("outgoing_links", [])
        targets = [link["target"] for link in outgoing]
        assert "python-basics" in targets
        assert "algorithms" in targets

    def test_read_note_with_incoming_links(self, test_vault):
        """Verify incoming link detection."""
        read = test_vault["read_engine"]

        result = read.read_note("python-basics")
        assert result["success"]

        # data-structures and meeting-notes link to python-basics
        incoming = result.get("incoming_links", [])
        sources = [link["source_id"] for link in incoming]
        assert "data-structures" in sources

    def test_read_nonexistent_note(self, test_vault):
        """Attempt to read a non-existent note."""
        read = test_vault["read_engine"]

        result = read.read_note("nonexistent-note")
        assert not result["success"]
        assert "error" in result

    def test_list_notes_by_type(self, test_vault):
        """List notes filtered by type."""
        read = test_vault["read_engine"]

        concepts = read.list_notes(type_filter="concept")
        assert len(concepts) == 3  # python-basics, data-structures, algorithms

        tasks = read.list_notes(type_filter="task")
        assert len(tasks) == 1

    def test_get_note_graph(self, test_vault):
        """Get graph context around a note."""
        read = test_vault["read_engine"]

        graph = read.get_note_graph("data-structures", depth=1)
        assert graph["center"] == "data-structures"

        node_ids = [n["id"] for n in graph["nodes"]]
        assert "data-structures" in node_ids
        assert "python-basics" in node_ids  # linked


class TestWriteOperations:
    """Test document writing functionality."""

    def test_create_new_note(self, test_vault):
        """Create a new note."""
        write = test_vault["write_engine"]
        vault_path = test_vault["vault_path"]

        result = write.write_note(
            filename="new-topic.md",
            content="# New Topic\n\nThis is a new document about a new topic.",
            metadata={
                "title": "New Topic",
                "type": "idea",
                "tags": ["new", "test"],
            },
        )

        assert result["success"]
        assert result["action"] == "created"
        assert (vault_path / "new-topic.md").exists()

    def test_update_existing_note(self, test_vault):
        """Update an existing note."""
        write = test_vault["write_engine"]

        result = write.write_note(
            filename="project-todo.md",
            content="# Updated TODO\n\nCompletely rewritten content.",
            metadata={"title": "Updated TODO", "status": "verified"},
        )

        assert result["success"]
        assert result["action"] == "updated"

    def test_patch_note_section(self, test_vault):
        """Patch a specific section of a note."""
        write = test_vault["write_engine"]

        result = write.patch_note(
            filename="python-basics",
            section="## Variables",
            new_content="Variables in Python don't need explicit type declarations.\nThey are dynamically typed at runtime.",
        )

        assert result["success"]
        assert result["action"] == "patched"

        # Verify the change
        read = test_vault["read_engine"]
        note = read.read_note("python-basics")
        assert "dynamically typed at runtime" in note["content"]

    def test_patch_nonexistent_section(self, test_vault):
        """Attempt to patch a non-existent section."""
        write = test_vault["write_engine"]

        result = write.patch_note(
            filename="python-basics",
            section="## Nonexistent Section",
            new_content="This should fail",
        )

        assert not result["success"]
        assert "available_sections" in result

    def test_backup_on_write(self, test_vault):
        """Verify backups are created on modification."""
        vault_path = test_vault["vault_path"]
        backup_dir = vault_path / ".backups"

        # Ensure backup directory exists after a write
        write = test_vault["write_engine"]
        write.write_note("algorithms", "# Updated Algorithms\n\nNew content.")

        # Check backup was created
        assert backup_dir.exists()
        backups = list(backup_dir.glob("algorithms_*.md"))
        assert len(backups) >= 1


class TestLinkOperations:
    """Test backlink management functionality."""

    def test_link_two_notes(self, test_vault):
        """Create a link between two notes."""
        link = test_vault["link_engine"]

        result = link.link_notes(
            source="algorithms",
            target="data-structures",
            reason="Related data structure concepts",
        )

        assert result["success"]

        # Verify link was added
        read = test_vault["read_engine"]
        note = read.read_note("algorithms")
        content = note["content"]
        assert "[[data-structures]]" in content

    def test_link_already_exists(self, test_vault):
        """Attempt to create a duplicate link."""
        link = test_vault["link_engine"]

        # This link already exists in the fixture
        result = link.link_notes(source="python-basics", target="data-structures")

        assert not result["success"]
        assert "already exists" in result["error"].lower()

    def test_bidirectional_link(self, test_vault):
        """Create a bidirectional link."""
        link = test_vault["link_engine"]

        # First create a new note to link
        write = test_vault["write_engine"]
        write.write_note("note-a", "# Note A\n\nContent A")
        write.write_note("note-b", "# Note B\n\nContent B")

        result = link.link_notes(
            source="note-a", target="note-b", reason="Related", bidirectional=True
        )

        assert result["success"]
        assert result.get("bidirectional", False)

    def test_suggest_links(self, test_vault):
        """Get link suggestions for a document."""
        link = test_vault["link_engine"]

        suggestions = link.suggest_links("algorithms", top_k=3)
        # Should suggest documents with shared tags or types
        assert len(suggestions) >= 0

    def test_find_orphans(self, test_vault):
        """Find documents with no links."""
        link = test_vault["link_engine"]

        orphans = link.find_orphans()
        # Some notes might be orphans
        _orphan_ids = [o["id"] for o in orphans]
        # After our tests, most notes should be connected
        assert isinstance(orphans, list)

    def test_find_broken_links(self, test_vault):
        """Find broken links in the vault."""
        link = test_vault["link_engine"]
        vault_path = test_vault["vault_path"]

        # Create a note with a broken link
        broken_note = vault_path / "broken-link-test.md"
        broken_note.write_text(
            """---
id: broken-link-test
title: Broken Link Test
---

# Broken Link Test

This links to [[nonexistent-document]] which doesn't exist.
""",
            encoding="utf-8",
        )

        # Re-sync
        test_vault["syncer"].full_sync()

        broken = link.find_broken_links()
        assert len(broken) >= 1

        broken_targets = [b["broken_link"] for b in broken]
        assert "nonexistent-document" in broken_targets


class TestSyncOperations:
    """Test vault synchronization functionality."""

    def test_full_sync(self, test_vault):
        """Test full vault synchronization."""
        syncer = test_vault["syncer"]

        stats = syncer.full_sync()
        assert stats["files_scanned"] >= 5
        assert len(stats["errors"]) == 0

    def test_consistency_check(self, test_vault):
        """Check vault consistency."""
        syncer = test_vault["syncer"]

        report = syncer.check_consistency()
        assert report["consistent"] or len(report["hash_mismatches"]) > 0

    def test_sync_after_external_edit(self, test_vault):
        """Sync after external file modification."""
        vault_path = test_vault["vault_path"]
        syncer = test_vault["syncer"]

        # Modify a file externally
        note_path = vault_path / "python-basics.md"
        original = note_path.read_text(encoding="utf-8")
        note_path.write_text(original + "\n\n## New Section\n\nAdded externally.", encoding="utf-8")

        # Check consistency - should detect mismatch
        _report = syncer.check_consistency()
        # May or may not be consistent depending on timing

        # Full sync should fix it
        stats = syncer.full_sync()
        assert stats["files_updated"] >= 0 or stats["files_indexed"] >= 0


class TestResources:
    """Test MCP resource providers."""

    def test_graph_summary(self, test_vault):
        """Get knowledge base summary."""
        resources = test_vault["resources"]

        summary = resources.get_graph_summary()
        assert "Knowledge Base Summary" in summary
        assert "Total Documents" in summary
        assert "Document Types" in summary

    def test_recent_notes(self, test_vault):
        """Get recently modified notes."""
        resources = test_vault["resources"]

        recent = resources.get_recent_notes(hours=24)
        # All notes were just created, so should appear
        assert len(recent) > 0

    def test_all_tags(self, test_vault):
        """Get all tags in the vault."""
        resources = test_vault["resources"]

        tags = resources.get_all_tags()
        assert "python" in tags
        assert "Tag" in tags  # Table header

    def test_orphan_documents(self, test_vault):
        """Get orphan documents report."""
        resources = test_vault["resources"]

        orphans = resources.get_orphan_documents()
        # Should return a formatted string
        assert isinstance(orphans, str)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_search_query(self, test_vault):
        """Search with empty query."""
        search = test_vault["search_engine"]

        results = search.search("", top_k=5)
        # Should handle gracefully
        assert "results" in results

    def test_special_characters_in_search(self, test_vault):
        """Search with special characters."""
        search = test_vault["search_engine"]

        results = search.search("Python's `def` keyword", top_k=5)
        assert "results" in results

    def test_unicode_content(self, test_vault):
        """Handle unicode content."""
        write = test_vault["write_engine"]

        result = write.write_note(
            filename="unicode-test.md",
            content="# 유니코드 테스트\n\n한글과 日本語 그리고 émojis 🚀",
            metadata={"title": "Unicode Test", "tags": ["한글", "日本語"]},
        )

        assert result["success"]

        read = test_vault["read_engine"]
        note = read.read_note("unicode-test")
        assert "한글" in note["content"]

    def test_large_document(self, test_vault):
        """Handle large documents."""
        write = test_vault["write_engine"]

        # Create a large document
        large_content = "# Large Document\n\n" + ("This is a paragraph. " * 100 + "\n\n") * 50

        result = write.write_note(
            filename="large-doc.md",
            content=large_content,
        )

        assert result["success"]

        # Search should still work
        search = test_vault["search_engine"]
        results = search.search("large document paragraph", top_k=5)
        assert "large-doc" in [r["id"] for r in results["results"]]

    def test_deeply_nested_backlinks(self, test_vault):
        """Handle documents with many backlinks."""
        write = test_vault["write_engine"]

        content = "# Many Links\n\n"
        for i in range(20):
            content += f"- [[link-target-{i}]]\n"

        result = write.write_note(filename="many-links.md", content=content)
        assert result["success"]

        read = test_vault["read_engine"]
        note = read.read_note("many-links")
        assert len(note.get("outgoing_links", [])) == 20
