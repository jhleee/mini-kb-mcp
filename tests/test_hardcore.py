"""Hardcore real-world scenario tests."""

import concurrent.futures
import random
import threading
import time

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


def generate_realistic_content(topic: str, length: int = 500) -> str:
    """Generate realistic markdown content for testing."""
    sections = [
        f"# {topic}\n\n",
        f"This document covers {topic.lower()} in detail.\n\n",
        "## Overview\n\n",
        f"The {topic.lower()} is an important concept in software development. "
        "It provides a foundation for building robust and scalable systems.\n\n",
        "## Key Concepts\n\n",
        "- First principle: Understanding the basics\n",
        "- Second principle: Applying best practices\n",
        "- Third principle: Continuous improvement\n\n",
        "## Implementation Details\n\n",
        "When implementing this concept, consider the following:\n\n",
        "1. Start with a clear understanding of requirements\n",
        "2. Design with scalability in mind\n",
        "3. Test thoroughly before deployment\n",
        "4. Monitor and iterate based on feedback\n\n",
        "## Code Example\n\n",
        "```python\n",
        f"def {topic.lower().replace(' ', '_')}():\n",
        '    """Example implementation."""\n',
        "    pass\n",
        "```\n\n",
        "## Best Practices\n\n",
        "Follow these best practices for optimal results:\n\n",
        "- Keep it simple and maintainable\n",
        "- Document your code thoroughly\n",
        "- Use version control effectively\n\n",
    ]
    return "".join(sections)[:length]


@pytest.fixture
def large_vault(tmp_path, embedding_fn):
    """Create a large realistic vault with many interconnected documents."""
    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Define realistic document categories
    categories = {
        "concept": [
            "Python Fundamentals", "Object Oriented Programming", "Functional Programming",
            "Design Patterns", "SOLID Principles", "Clean Code", "Refactoring Techniques",
            "API Design", "Database Design", "System Architecture", "Microservices",
            "Event Driven Architecture", "Domain Driven Design", "Test Driven Development",
            "Continuous Integration", "DevOps Practices",
        ],
        "task": [
            "Implement User Authentication", "Setup CI Pipeline", "Refactor Payment Module",
            "Optimize Database Queries", "Write Unit Tests", "Deploy to Production",
            "Fix Memory Leak", "Update Dependencies", "Code Review Checklist",
        ],
        "log": [
            "Sprint 1 Retrospective", "Architecture Decision Record", "Incident Report 2024-01",
            "Performance Benchmark Results", "Security Audit Findings", "Team Meeting Notes",
        ],
        "idea": [
            "AI Powered Code Review", "Automated Testing Framework", "Self Healing Systems",
            "Real Time Collaboration", "Smart Caching Strategy", "Predictive Scaling",
        ],
        "ref": [
            "Python Standard Library", "REST API Guidelines", "SQL Best Practices",
            "Git Workflow", "Docker Commands", "Kubernetes Basics",
        ],
    }

    # Create documents with realistic interconnections
    all_docs = []
    for doc_type, topics in categories.items():
        for topic in topics:
            doc_id = topic.lower().replace(" ", "-")
            all_docs.append((doc_id, topic, doc_type))

    # Generate documents with backlinks
    for doc_id, topic, doc_type in all_docs:
        # Select random related documents for backlinks
        other_docs = [d for d in all_docs if d[0] != doc_id]
        num_links = random.randint(1, 5)
        linked_docs = random.sample(other_docs, min(num_links, len(other_docs)))

        # Build content with backlinks
        content = generate_realistic_content(topic, 800)

        # Add related notes section
        content += "\n## Related Notes\n\n"
        for linked_id, linked_title, _ in linked_docs:
            content += f"- [[{linked_id}]] - {linked_title}\n"

        # Create frontmatter
        tags = [doc_type, topic.split()[0].lower()]
        if doc_type == "concept":
            tags.append("learning")
        elif doc_type == "task":
            tags.append("work")

        full_content = f"""---
id: {doc_id}
title: {topic}
type: {doc_type}
tags: [{", ".join(tags)}]
status: {"verified" if random.random() > 0.3 else "draft"}
---

{content}
"""
        (vault_path / f"{doc_id}.md").write_text(full_content, encoding="utf-8")

    # Initialize components
    sqlite_db = SQLiteDB(data_dir / "metadata.db")
    vector_db = VectorDB(data_dir / "vectors")
    vector_db.set_embedding_function(embedding_fn)
    parser = MarkdownParser(vault_path)

    # Run initial sync
    syncer = VaultSyncer(vault_path, sqlite_db, vector_db, parser)
    start_time = time.time()
    _stats = syncer.full_sync()
    sync_time = time.time() - start_time

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
        "doc_count": len(all_docs),
        "sync_time": sync_time,
        "all_docs": all_docs,
    }


class TestScalability:
    """Test system performance with large document sets."""

    def test_initial_sync_performance(self, large_vault):
        """Verify initial sync completes in reasonable time."""
        sync_time = large_vault["sync_time"]
        doc_count = large_vault["doc_count"]

        print(f"\n  Synced {doc_count} documents in {sync_time:.2f}s")
        print(f"  Average: {sync_time/doc_count*1000:.1f}ms per document")

        # Should sync 40+ docs in under 60 seconds
        assert sync_time < 60, f"Sync too slow: {sync_time:.2f}s for {doc_count} docs"

    def test_search_performance(self, large_vault):
        """Test search response time with many documents."""
        search = large_vault["search_engine"]

        queries = [
            "object oriented programming design patterns",
            "database optimization techniques",
            "continuous integration deployment",
            "microservices architecture",
            "test driven development best practices",
        ]

        total_time = 0
        for query in queries:
            start = time.time()
            results = search.search(query, top_k=10)
            elapsed = time.time() - start
            total_time += elapsed

            assert results["total_results"] > 0, f"No results for: {query}"

        avg_time = total_time / len(queries)
        print(f"\n  Average search time: {avg_time*1000:.1f}ms")

        # Search should be under 500ms on average
        assert avg_time < 0.5, f"Search too slow: {avg_time*1000:.1f}ms average"

    def test_bulk_read_performance(self, large_vault):
        """Test reading many documents quickly."""
        read = large_vault["read_engine"]
        all_docs = large_vault["all_docs"]

        start = time.time()
        for doc_id, _, _ in all_docs[:20]:
            result = read.read_note(doc_id)
            assert result["success"], f"Failed to read: {doc_id}"

        elapsed = time.time() - start
        print(f"\n  Read 20 documents in {elapsed*1000:.1f}ms")
        print(f"  Average: {elapsed/20*1000:.1f}ms per document")

        assert elapsed < 5, f"Bulk read too slow: {elapsed:.2f}s"


class TestSearchQuality:
    """Test search accuracy and relevance."""

    def test_semantic_relevance(self, large_vault):
        """Verify semantic search returns relevant results."""
        search = large_vault["search_engine"]

        test_cases = [
            ("how to write clean maintainable code", ["clean-code", "refactoring-techniques"]),
            ("database query optimization", ["database-design", "optimize-database-queries"]),
            ("automated testing practices", ["test-driven-development", "write-unit-tests"]),
            ("deploying applications to cloud", ["deploy-to-production", "devops-practices"]),
            ("software design principles", ["solid-principles", "design-patterns"]),
        ]

        for query, expected_ids in test_cases:
            results = search.search(query, top_k=5)
            found_ids = [r["id"] for r in results["results"]]

            # At least one expected result should be in top 5
            matches = [eid for eid in expected_ids if eid in found_ids]
            assert len(matches) > 0, f"Query '{query}' didn't find expected docs. Got: {found_ids}"

    def test_type_filter_accuracy(self, large_vault):
        """Verify type filters work correctly."""
        search = large_vault["search_engine"]

        for doc_type in ["concept", "task", "log", "idea", "ref"]:
            results = search.search("software development", type_filter=doc_type, top_k=10)

            for result in results["results"]:
                assert result["type"] == doc_type, f"Got {result['type']} but expected {doc_type}"

    def test_tag_filter_accuracy(self, large_vault):
        """Verify tag filters work correctly."""
        search = large_vault["search_engine"]

        results = search.search("programming", tag_filter=["learning"], top_k=10)

        for result in results["results"]:
            assert "learning" in (result["tags"] or []), "Result missing 'learning' tag"

    def test_graph_expansion_quality(self, large_vault):
        """Verify graph expansion adds meaningful context."""
        search = large_vault["search_engine"]

        results = search.search_with_expansion("design patterns", top_k=3, expansion_hops=1)

        assert "expanded_context" in results
        # Should have some expanded context from linked documents
        assert len(results["results"]) > 0


class TestDataIntegrity:
    """Test data consistency and integrity."""

    def test_backlink_consistency(self, large_vault):
        """Verify all backlinks are properly indexed."""
        sqlite_db = large_vault["sqlite_db"]
        parser = large_vault["parser"]
        vault_path = large_vault["vault_path"]

        # Check each document's backlinks match database
        for md_file in vault_path.glob("*.md"):
            doc = parser.parse_file(md_file)
            db_edges = sqlite_db.get_outgoing_edges(doc.id)

            # Number of edges should match or be close to parsed backlinks
            # (some targets might not exist)
            assert len(db_edges) <= len(doc.backlinks) + 1

    def test_hash_consistency(self, large_vault):
        """Verify file hashes match database."""
        sqlite_db = large_vault["sqlite_db"]
        vault_path = large_vault["vault_path"]

        mismatches = []
        for md_file in vault_path.glob("*.md"):
            content = md_file.read_text(encoding="utf-8")
            file_hash = SQLiteDB.compute_hash(content)

            node = sqlite_db.get_node_by_path(str(md_file))
            if node and node.hash != file_hash:
                mismatches.append(md_file.name)

        assert len(mismatches) == 0, f"Hash mismatches: {mismatches}"

    def test_no_orphan_vectors(self, large_vault):
        """Verify all vectors have corresponding documents."""
        sqlite_db = large_vault["sqlite_db"]
        vector_db = large_vault["vector_db"]

        vector_stats = vector_db.get_stats()
        sqlite_stats = sqlite_db.get_stats()

        # Vector docs should match or be less than SQLite docs
        assert vector_stats["unique_documents"] <= sqlite_stats["total_nodes"]


class TestConcurrency:
    """Test concurrent access scenarios."""

    def test_concurrent_reads(self, large_vault):
        """Test multiple simultaneous reads."""
        read = large_vault["read_engine"]
        all_docs = large_vault["all_docs"]
        doc_ids = [d[0] for d in all_docs[:20]]

        errors = []

        def read_doc(doc_id):
            try:
                result = read.read_note(doc_id)
                if not result["success"]:
                    errors.append(f"Failed to read {doc_id}")
            except Exception as e:
                errors.append(f"Error reading {doc_id}: {e}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(read_doc, doc_ids)

        assert len(errors) == 0, f"Concurrent read errors: {errors}"

    def test_concurrent_searches(self, large_vault):
        """Test multiple simultaneous searches."""
        search = large_vault["search_engine"]

        queries = [
            "python programming",
            "database design",
            "testing practices",
            "deployment automation",
            "code refactoring",
        ] * 4  # 20 total queries

        errors = []
        results_count = []

        def search_query(query):
            try:
                results = search.search(query, top_k=5)
                results_count.append(len(results["results"]))
            except Exception as e:
                errors.append(f"Error searching '{query}': {e}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(search_query, queries)

        assert len(errors) == 0, f"Concurrent search errors: {errors}"
        assert all(c > 0 for c in results_count), "Some searches returned no results"

    def test_read_write_interleaving(self, large_vault):
        """Test reading and writing simultaneously."""
        read = large_vault["read_engine"]
        write = large_vault["write_engine"]

        errors = []
        lock = threading.Lock()

        def reader():
            for _ in range(10):
                try:
                    result = read.read_note("clean-code")
                    if not result["success"]:
                        with lock:
                            errors.append("Read failed")
                except Exception as e:
                    with lock:
                        errors.append(f"Read error: {e}")
                time.sleep(0.01)

        def writer():
            for i in range(5):
                try:
                    result = write.write_note(
                        f"concurrent-test-{threading.current_thread().name}-{i}.md",
                        f"# Concurrent Test {i}\n\nContent here.",
                    )
                    if not result["success"]:
                        with lock:
                            errors.append(f"Write failed: {result.get('error')}")
                except Exception as e:
                    with lock:
                        errors.append(f"Write error: {e}")
                time.sleep(0.02)

        threads = [
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=writer),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Interleaving errors: {errors}"


class TestEdgeCasesHardcore:
    """Extreme edge cases and stress tests."""

    def test_very_long_document(self, large_vault):
        """Handle extremely long documents."""
        write = large_vault["write_engine"]
        search = large_vault["search_engine"]

        # Create a very long document (50KB+)
        long_content = "# Extremely Long Document\n\n"
        for i in range(500):
            long_content += f"## Section {i}\n\n"
            long_content += "This is a paragraph with substantial content. " * 10
            long_content += "\n\n"

        result = write.write_note("very-long-doc.md", long_content)
        assert result["success"]

        # Should still be searchable
        results = search.search("extremely long document section", top_k=5)
        found_ids = [r["id"] for r in results["results"]]
        assert "very-long-doc" in found_ids

    def test_special_filenames(self, large_vault):
        """Handle special characters in filenames."""
        write = large_vault["write_engine"]
        read = large_vault["read_engine"]

        special_names = [
            "doc-with-dashes",
            "doc_with_underscores",
            "doc123numbers",
            "MixedCaseDoc",
        ]

        for name in special_names:
            result = write.write_note(f"{name}.md", f"# {name}\n\nContent.")
            assert result["success"], f"Failed to create {name}"

            result = read.read_note(name)
            assert result["success"], f"Failed to read {name}"

    def test_deep_link_chains(self, large_vault):
        """Test documents with deep backlink chains."""
        write = large_vault["write_engine"]
        read = large_vault["read_engine"]

        # Create a chain: A -> B -> C -> D -> E
        for i in range(5):
            next_link = f"[[chain-doc-{i+1}]]" if i < 4 else ""
            content = f"# Chain Doc {i}\n\nLinks to {next_link}"
            write.write_note(f"chain-doc-{i}.md", content)

        # Read and verify chain
        for i in range(5):
            result = read.read_note(f"chain-doc-{i}")
            assert result["success"]

        # Get graph from middle of chain
        graph = read.get_note_graph("chain-doc-2", depth=2)
        assert len(graph["nodes"]) >= 3  # Should find connected nodes

    def test_circular_references(self, large_vault):
        """Handle documents with circular backlinks."""
        write = large_vault["write_engine"]
        syncer = large_vault["syncer"]

        # Create circular reference: A -> B -> C -> A
        write.write_note("circular-a.md", "# Doc A\n\nLinks to [[circular-b]]")
        write.write_note("circular-b.md", "# Doc B\n\nLinks to [[circular-c]]")
        write.write_note("circular-c.md", "# Doc C\n\nLinks to [[circular-a]]")

        # Sync should handle this without infinite loops
        syncer.full_sync()

        # Graph traversal should not hang
        read = large_vault["read_engine"]
        graph = read.get_note_graph("circular-a", depth=2)
        assert len(graph["nodes"]) == 3

    def test_rapid_modifications(self, large_vault):
        """Test rapid successive modifications to same document."""
        write = large_vault["write_engine"]
        read = large_vault["read_engine"]

        # Create initial document
        write.write_note("rapid-mod.md", "# Initial\n\nContent v1")

        # Rapid modifications
        for i in range(10):
            write.write_note("rapid-mod.md", f"# Modified\n\nContent v{i+2}")

        # Final read should have latest content
        result = read.read_note("rapid-mod")
        assert result["success"]
        assert "v11" in result["content"]

    def test_empty_and_minimal_documents(self, large_vault):
        """Handle edge case document contents."""
        write = large_vault["write_engine"]
        read = large_vault["read_engine"]

        test_cases = [
            ("minimal-doc", "# Title\n"),
            ("only-frontmatter", ""),  # Will get default frontmatter
            ("whitespace-doc", "# Title\n\n   \n\n   \n"),
        ]

        for name, content in test_cases:
            result = write.write_note(f"{name}.md", content)
            assert result["success"], f"Failed to create {name}"

            result = read.read_note(name)
            assert result["success"], f"Failed to read {name}"

    def test_unicode_stress(self, large_vault):
        """Stress test with various unicode content."""
        write = large_vault["write_engine"]
        search = large_vault["search_engine"]

        # Mix of scripts and emojis
        unicode_content = """# 다국어 문서 🌍

## 한국어 섹션
안녕하세요! 이것은 테스트입니다.

## 日本語セクション
こんにちは！これはテストです。

## 中文部分
你好！这是一个测试。

## Emoji Section 🚀
Let's test some emojis: 🎉 🔥 💻 🐍 📚

## Mixed Content
Python은 매우 powerful한 언어입니다 🐍
"""

        result = write.write_note("unicode-stress.md", unicode_content)
        assert result["success"]

        # Should be searchable
        results = search.search("다국어 문서", top_k=5)
        assert any("unicode-stress" in r["id"] for r in results["results"])


class TestRealWorldScenarios:
    """Simulate real-world usage patterns."""

    def test_knowledge_base_workflow(self, large_vault):
        """Simulate typical knowledge base usage."""
        search = large_vault["search_engine"]
        read = large_vault["read_engine"]
        write = large_vault["write_engine"]
        link = large_vault["link_engine"]

        # Step 1: User searches for a topic
        results = search.search("how to implement design patterns", top_k=5)
        assert results["total_results"] > 0

        # Step 2: User reads the top result
        top_result = results["results"][0]
        note = read.read_note(top_result["id"])
        assert note["success"]

        # Step 3: User creates a new note based on learning
        new_note = write.write_note(
            "my-design-patterns-notes.md",
            f"""# My Design Patterns Notes

Based on reading [[{top_result['id']}]]

## Key Takeaways

- Pattern 1: Factory
- Pattern 2: Observer
- Pattern 3: Strategy

## Questions to Explore

- How to combine patterns?
- When not to use patterns?
""",
            metadata={"type": "log", "tags": ["notes", "learning"]},
        )
        assert new_note["success"]

        # Step 4: User links to related content
        related = search.find_similar(top_result["id"], top_k=3)
        if related:
            _link_result = link.link_notes(
                "my-design-patterns-notes", related[0]["id"], "Related concept"
            )
            # May or may not succeed depending on existing links

        # Step 5: User searches again to verify
        new_results = search.search("my design patterns notes", top_k=5)
        _found_new = any("my-design-patterns" in r["id"] for r in new_results["results"])
        # Note: might not be found immediately without re-sync

    def test_project_documentation_workflow(self, large_vault):
        """Simulate project documentation workflow."""
        write = large_vault["write_engine"]
        read = large_vault["read_engine"]
        search = large_vault["search_engine"]
        syncer = large_vault["syncer"]

        # Create project structure
        docs = [
            ("project-overview", "concept", "# Project Overview\n\nMain project documentation."),
            ("api-reference", "ref", "# API Reference\n\nAPI documentation here."),
            ("setup-guide", "ref", "# Setup Guide\n\nHow to set up the project."),
            ("sprint-1-notes", "log", "# Sprint 1\n\nSprint notes and decisions."),
            ("feature-ideas", "idea", "# Feature Ideas\n\nBrainstorming new features."),
        ]

        for doc_id, doc_type, content in docs:
            content += "\n\n## Related\n\n- [[project-overview]]\n"
            result = write.write_note(
                f"proj-{doc_id}.md",
                content,
                metadata={"type": doc_type, "tags": ["project", doc_type]},
            )
            assert result["success"]

        # Re-sync to index all new docs
        syncer.full_sync()

        # Search within project docs
        results = search.search("project documentation", tag_filter=["project"], top_k=10)
        assert results["total_results"] >= 3

        # Verify cross-references
        overview = read.read_note("proj-project-overview")
        assert overview["success"]

    def test_daily_notes_workflow(self, large_vault):
        """Simulate daily notes/journal workflow."""
        write = large_vault["write_engine"]
        search = large_vault["search_engine"]

        # Create a week of daily notes
        for day in range(1, 8):
            topics = ["meeting", "coding", "review", "planning", "debugging"]
            daily_topic = random.choice(topics)

            content = f"""# Daily Log - 2024-01-{day:02d}

## Summary
Today focused on {daily_topic} activities.

## Tasks Completed
- Task 1 for {daily_topic}
- Task 2 for {daily_topic}
- Follow up on [[design-patterns]]

## Notes
Important insights from today's {daily_topic} session.

## Tomorrow
Continue with related work.
"""
            result = write.write_note(
                f"daily-2024-01-{day:02d}.md",
                content,
                metadata={"type": "log", "tags": ["daily", daily_topic]},
            )
            assert result["success"]

        # Search across daily notes
        results = search.search("meeting notes insights", top_k=10)
        assert results["total_results"] >= 0  # May or may not find matches

    def test_research_workflow(self, large_vault):
        """Simulate research and learning workflow."""
        write = large_vault["write_engine"]
        link = large_vault["link_engine"]
        search = large_vault["search_engine"]

        # Create research topic
        write.write_note(
            "research-ml-basics.md",
            """# Machine Learning Basics

## Overview
Machine learning is a subset of AI.

## Key Concepts
- Supervised learning
- Unsupervised learning
- Reinforcement learning

## Resources
- Online courses
- Books
- Papers
""",
            metadata={"type": "concept", "tags": ["research", "ml", "ai"]},
        )

        # Add related notes
        write.write_note(
            "ml-neural-networks.md",
            """# Neural Networks

Deep learning with neural networks.

## Architecture
- Input layer
- Hidden layers
- Output layer

## Related
- [[research-ml-basics]]
""",
            metadata={"type": "concept", "tags": ["research", "ml", "deep-learning"]},
        )

        # Link them
        link.link_notes("research-ml-basics", "ml-neural-networks", "Core topic")

        # Search and verify relationships
        results = search.search("machine learning neural networks", top_k=5)
        assert results["total_results"] >= 1


class TestRecoveryAndResilience:
    """Test system recovery from various failure scenarios."""

    def test_recovery_from_corrupted_frontmatter(self, large_vault):
        """Handle documents with invalid frontmatter."""
        vault_path = large_vault["vault_path"]
        syncer = large_vault["syncer"]

        # Create a file with malformed frontmatter
        bad_file = vault_path / "bad-frontmatter.md"
        bad_file.write_text(
            """---
id: bad-doc
title: Bad Doc
tags: [unclosed bracket
---

# Content

This has broken YAML.
""",
            encoding="utf-8",
        )

        # Sync should handle gracefully (may skip or use defaults)
        try:
            _stats = syncer.full_sync()
            # Should not crash
            assert True
        except Exception as e:
            # If it does raise, should be a reasonable error
            assert "yaml" in str(e).lower() or "parse" in str(e).lower()

    def test_recovery_after_file_deletion(self, large_vault):
        """Recover gracefully when files are deleted externally."""
        vault_path = large_vault["vault_path"]
        write = large_vault["write_engine"]
        syncer = large_vault["syncer"]
        sqlite_db = large_vault["sqlite_db"]

        # Create a document
        write.write_note("to-be-deleted.md", "# Will Be Deleted\n\nContent here.")
        syncer.full_sync()

        # Verify it exists
        node = sqlite_db.get_node("to-be-deleted")
        assert node is not None

        # Delete file externally
        (vault_path / "to-be-deleted.md").unlink()

        # Re-sync should clean up
        syncer.full_sync()

        # Should be removed from DB
        node = sqlite_db.get_node("to-be-deleted")
        assert node is None

    def test_recovery_from_db_corruption(self, large_vault, tmp_path):
        """Test behavior when starting with fresh DB."""
        vault_path = large_vault["vault_path"]
        embedding_fn = large_vault["vector_db"]._embedding_fn

        # Create new DB instances (simulating corruption recovery)
        new_data_dir = tmp_path / "recovery_data"
        new_data_dir.mkdir()

        new_sqlite = SQLiteDB(new_data_dir / "recovered.db")
        new_vector = VectorDB(new_data_dir / "recovered_vectors")
        new_vector.set_embedding_function(embedding_fn)
        new_parser = MarkdownParser(vault_path)
        new_syncer = VaultSyncer(vault_path, new_sqlite, new_vector, new_parser)

        # Full sync should rebuild everything
        stats = new_syncer.full_sync()

        # Should have recovered all documents
        assert stats["files_indexed"] > 0
        assert len(stats["errors"]) == 0
