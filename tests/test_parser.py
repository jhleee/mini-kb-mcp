"""Tests for the markdown parser."""


import pytest

from mini_docs_mcp.parser.markdown import MarkdownParser


@pytest.fixture
def temp_vault(tmp_path):
    """Create a temporary vault with test files."""
    vault_path = tmp_path / "vault"
    vault_path.mkdir()

    # Create test files
    test_note = vault_path / "test-note.md"
    test_note.write_text(
        """---
id: test-note
title: Test Note
type: concept
tags: [testing, example]
status: draft
---

# Test Note

This is a test document with a [[backlink]] to another note.

## Section One

Content in section one.

## Section Two

Content with [[another-link|display text]] here.
""",
        encoding="utf-8",
    )

    linked_note = vault_path / "backlink.md"
    linked_note.write_text(
        """---
id: backlink
title: Backlink Note
type: ref
---

# Backlink Note

This note is linked from [[test-note]].
""",
        encoding="utf-8",
    )

    return vault_path


class TestMarkdownParser:
    """Tests for MarkdownParser."""

    def test_parse_file(self, temp_vault):
        """Test parsing a markdown file."""
        parser = MarkdownParser(temp_vault)
        doc = parser.parse_file(temp_vault / "test-note.md")

        assert doc.id == "test-note"
        assert doc.title == "Test Note"
        assert doc.type == "concept"
        assert doc.status == "draft"
        assert "testing" in doc.tags
        assert "example" in doc.tags

    def test_extract_backlinks(self, temp_vault):
        """Test backlink extraction."""
        parser = MarkdownParser(temp_vault)
        doc = parser.parse_file(temp_vault / "test-note.md")

        assert len(doc.backlinks) == 2
        targets = [bl.target for bl in doc.backlinks]
        assert "backlink" in targets
        assert "another-link" in targets

    def test_get_sections(self, temp_vault):
        """Test section extraction."""
        parser = MarkdownParser(temp_vault)
        content = (temp_vault / "test-note.md").read_text(encoding="utf-8")
        sections = parser.get_sections(content)

        titles = [s["title"] for s in sections]
        assert "Test Note" in titles
        assert "Section One" in titles
        assert "Section Two" in titles

    def test_find_section(self, temp_vault):
        """Test finding a specific section."""
        parser = MarkdownParser(temp_vault)
        content = (temp_vault / "test-note.md").read_text(encoding="utf-8")

        section = parser.find_section(content, "## Section One")
        assert section is not None
        assert section["title"] == "Section One"

    def test_replace_section(self, temp_vault):
        """Test replacing a section."""
        parser = MarkdownParser(temp_vault)
        content = (temp_vault / "test-note.md").read_text(encoding="utf-8")

        new_content = parser.replace_section(
            content, "## Section One", "Updated content here."
        )
        assert new_content is not None
        assert "Updated content here." in new_content
        assert "Content in section one." not in new_content

    def test_resolve_link(self, temp_vault):
        """Test link resolution."""
        parser = MarkdownParser(temp_vault)

        path = parser.resolve_link("backlink")
        assert path is not None
        assert path.name == "backlink.md"

        path = parser.resolve_link("nonexistent")
        assert path is None

    def test_get_summary(self, temp_vault):
        """Test summary generation."""
        parser = MarkdownParser(temp_vault)
        doc = parser.parse_file(temp_vault / "test-note.md")

        summary = doc.get_summary(100)
        assert len(summary) <= 103  # 100 + "..."
        assert "test document" in summary.lower()

    def test_add_backlink(self, temp_vault):
        """Test adding a backlink."""
        parser = MarkdownParser(temp_vault)
        content = (temp_vault / "test-note.md").read_text(encoding="utf-8")

        updated = parser.add_backlink(content, "new-target", "Related concept")
        assert "[[new-target]]" in updated
        assert "Related concept" in updated

    def test_scan_vault(self, temp_vault):
        """Test vault scanning."""
        parser = MarkdownParser(temp_vault)
        files = parser.scan_vault()

        assert len(files) == 2
        names = [f.name for f in files]
        assert "test-note.md" in names
        assert "backlink.md" in names
