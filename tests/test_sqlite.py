"""Tests for the SQLite database module."""


import pytest

from mini_docs_mcp.db.sqlite import Edge, Node, SQLiteDB


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database."""
    db_path = tmp_path / "test.db"
    return SQLiteDB(db_path)


class TestSQLiteDB:
    """Tests for SQLiteDB."""

    def test_create_database(self, temp_db):
        """Test database creation."""
        assert temp_db.db_path.exists()

    def test_upsert_and_get_node(self, temp_db):
        """Test node insertion and retrieval."""
        node = Node(
            id="test-node",
            file_path="/path/to/test.md",
            title="Test Node",
            type="concept",
            tags=["test", "example"],
            hash="abc123",
            summary="A test node",
        )
        temp_db.upsert_node(node)

        retrieved = temp_db.get_node("test-node")
        assert retrieved is not None
        assert retrieved.title == "Test Node"
        assert retrieved.type == "concept"
        assert "test" in retrieved.tags

    def test_update_node(self, temp_db):
        """Test node update."""
        node = Node(
            id="update-test",
            file_path="/path/to/update.md",
            title="Original Title",
        )
        temp_db.upsert_node(node)

        node.title = "Updated Title"
        temp_db.upsert_node(node)

        retrieved = temp_db.get_node("update-test")
        assert retrieved.title == "Updated Title"

    def test_get_node_by_path(self, temp_db):
        """Test getting node by file path."""
        node = Node(
            id="path-test",
            file_path="/unique/path/test.md",
            title="Path Test",
        )
        temp_db.upsert_node(node)

        retrieved = temp_db.get_node_by_path("/unique/path/test.md")
        assert retrieved is not None
        assert retrieved.id == "path-test"

    def test_get_all_nodes(self, temp_db):
        """Test getting all nodes."""
        for i in range(3):
            node = Node(
                id=f"node-{i}",
                file_path=f"/path/node-{i}.md",
                title=f"Node {i}",
            )
            temp_db.upsert_node(node)

        nodes = temp_db.get_all_nodes()
        assert len(nodes) == 3

    def test_get_nodes_by_type(self, temp_db):
        """Test filtering nodes by type."""
        temp_db.upsert_node(Node(id="c1", file_path="/c1.md", type="concept"))
        temp_db.upsert_node(Node(id="c2", file_path="/c2.md", type="concept"))
        temp_db.upsert_node(Node(id="t1", file_path="/t1.md", type="task"))

        concepts = temp_db.get_nodes_by_type("concept")
        assert len(concepts) == 2

        tasks = temp_db.get_nodes_by_type("task")
        assert len(tasks) == 1

    def test_delete_node(self, temp_db):
        """Test node deletion."""
        node = Node(id="delete-me", file_path="/delete.md")
        temp_db.upsert_node(node)

        assert temp_db.delete_node("delete-me")
        assert temp_db.get_node("delete-me") is None

    def test_edge_operations(self, temp_db):
        """Test edge (backlink) operations."""
        # Create nodes first
        temp_db.upsert_node(Node(id="source", file_path="/source.md"))
        temp_db.upsert_node(Node(id="target", file_path="/target.md"))

        # Create edge
        edge = Edge(
            id=None,
            source_id="source",
            target_id="target",
            context_snippet="Links to target",
        )
        temp_db.upsert_edge(edge)

        # Get outgoing edges
        outgoing = temp_db.get_outgoing_edges("source")
        assert len(outgoing) == 1
        assert outgoing[0].target_id == "target"

        # Get incoming edges
        incoming = temp_db.get_incoming_edges("target")
        assert len(incoming) == 1
        assert incoming[0].source_id == "source"

    def test_delete_edges_for_source(self, temp_db):
        """Test deleting edges for a source node."""
        temp_db.upsert_node(Node(id="s1", file_path="/s1.md"))
        temp_db.upsert_node(Node(id="t1", file_path="/t1.md"))
        temp_db.upsert_node(Node(id="t2", file_path="/t2.md"))

        temp_db.upsert_edge(Edge(id=None, source_id="s1", target_id="t1"))
        temp_db.upsert_edge(Edge(id=None, source_id="s1", target_id="t2"))

        deleted = temp_db.delete_edges_for_source("s1")
        assert deleted == 2

        outgoing = temp_db.get_outgoing_edges("s1")
        assert len(outgoing) == 0

    def test_get_stats(self, temp_db):
        """Test statistics retrieval."""
        # Add some nodes and edges
        temp_db.upsert_node(Node(id="n1", file_path="/n1.md", type="concept"))
        temp_db.upsert_node(Node(id="n2", file_path="/n2.md", type="concept"))
        temp_db.upsert_node(Node(id="n3", file_path="/n3.md", type="idea"))
        temp_db.upsert_edge(Edge(id=None, source_id="n1", target_id="n2"))

        stats = temp_db.get_stats()
        assert stats["total_nodes"] == 3
        assert stats["total_edges"] == 1
        assert stats["type_distribution"]["concept"] == 2
        assert stats["type_distribution"]["idea"] == 1

    def test_get_all_tags(self, temp_db):
        """Test tag aggregation."""
        temp_db.upsert_node(
            Node(id="t1", file_path="/t1.md", tags=["python", "testing"])
        )
        temp_db.upsert_node(
            Node(id="t2", file_path="/t2.md", tags=["python", "database"])
        )

        tags = temp_db.get_all_tags()
        assert tags["python"] == 2
        assert tags["testing"] == 1
        assert tags["database"] == 1

    def test_compute_hash(self):
        """Test content hashing."""
        content = "Test content"
        hash1 = SQLiteDB.compute_hash(content)
        hash2 = SQLiteDB.compute_hash(content)
        hash3 = SQLiteDB.compute_hash("Different content")

        assert hash1 == hash2
        assert hash1 != hash3
