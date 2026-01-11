"""Markdown parser for frontmatter and backlinks."""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import frontmatter

logger = logging.getLogger(__name__)

# Backlink pattern: [[Note Name]] or [[Note Name|Display Text]]
BACKLINK_PATTERN = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")

# Section header pattern for patch operations
SECTION_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


@dataclass
class BacklinkMatch:
    """Represents a backlink found in a document."""

    target: str  # The linked note name
    context: str  # Surrounding context (sentence or line)
    position: int  # Character position in the document


@dataclass
class ParsedDocument:
    """Represents a fully parsed markdown document."""

    file_path: str
    id: str
    title: str | None
    type: str
    status: str
    tags: list[str]
    content: str  # Full content including frontmatter
    body: str  # Content without frontmatter
    backlinks: list[BacklinkMatch]
    raw_metadata: dict

    def get_summary(self, max_length: int = 200) -> str:
        """Generate a summary from the document body."""
        # Remove headers and links for cleaner summary
        text = re.sub(r"^#+\s+.+$", "", self.body, flags=re.MULTILINE)
        text = re.sub(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", r"\1", text)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) > max_length:
            return text[:max_length].rsplit(" ", 1)[0] + "..."
        return text


class MarkdownParser:
    """Parser for markdown files with frontmatter and backlinks."""

    def __init__(self, vault_path: Path | str):
        """Initialize the parser.

        Args:
            vault_path: Root path of the markdown vault.
        """
        self.vault_path = Path(vault_path)

    def parse_file(self, file_path: Path | str) -> ParsedDocument:
        """Parse a markdown file.

        Args:
            file_path: Path to the markdown file.

        Returns:
            ParsedDocument with extracted metadata and backlinks.
        """
        file_path = Path(file_path)
        content = file_path.read_text(encoding="utf-8")
        return self.parse_content(content, str(file_path))

    def parse_content(self, content: str, file_path: str) -> ParsedDocument:
        """Parse markdown content.

        Args:
            content: Raw markdown content.
            file_path: Path to the file (for metadata).

        Returns:
            ParsedDocument with extracted metadata and backlinks.
        """
        # Parse frontmatter
        post = frontmatter.loads(content)
        metadata = dict(post.metadata)
        body = post.content

        # Extract ID from metadata or generate from filename
        path = Path(file_path)
        doc_id = metadata.get("id", path.stem)

        # Extract other metadata with defaults
        title = metadata.get("title", path.stem.replace("-", " ").title())
        doc_type = metadata.get("type", "note")
        status = metadata.get("status", "draft")
        tags = metadata.get("tags", [])
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]

        # Extract backlinks with context
        backlinks = self._extract_backlinks(body)

        return ParsedDocument(
            file_path=file_path,
            id=doc_id,
            title=title,
            type=doc_type,
            status=status,
            tags=tags,
            content=content,
            body=body,
            backlinks=backlinks,
            raw_metadata=metadata,
        )

    def _extract_backlinks(self, text: str) -> list[BacklinkMatch]:
        """Extract all backlinks from text with context.

        Args:
            text: Document body text.

        Returns:
            List of BacklinkMatch objects.
        """
        backlinks = []
        lines = text.split("\n")
        current_pos = 0

        for line in lines:
            for match in BACKLINK_PATTERN.finditer(line):
                target = match.group(1).strip()
                # Get context: the entire line containing the backlink
                context = line.strip()
                # Limit context length
                if len(context) > 150:
                    # Try to get context around the match
                    start = max(0, match.start() - 50)
                    end = min(len(line), match.end() + 50)
                    context = "..." + line[start:end].strip() + "..."

                backlinks.append(
                    BacklinkMatch(
                        target=target,
                        context=context,
                        position=current_pos + match.start(),
                    )
                )
            current_pos += len(line) + 1  # +1 for newline

        return backlinks

    def get_sections(self, content: str) -> list[dict]:
        """Get all sections from a markdown document.

        Args:
            content: Markdown content.

        Returns:
            List of sections with level, title, start, and end positions.
        """
        post = frontmatter.loads(content)
        body = post.content

        sections = []
        matches = list(SECTION_PATTERN.finditer(body))

        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()
            start = match.start()

            # End is either the start of next section or end of document
            end = matches[i + 1].start() if i + 1 < len(matches) else len(body)

            sections.append(
                {
                    "level": level,
                    "title": title,
                    "header": match.group(0),
                    "start": start,
                    "end": end,
                    "content": body[start:end].strip(),
                }
            )

        return sections

    def find_section(self, content: str, section_header: str) -> dict | None:
        """Find a specific section by its header.

        Args:
            content: Markdown content.
            section_header: Section header to find (e.g., "## Overview").

        Returns:
            Section dict or None if not found.
        """
        sections = self.get_sections(content)
        # Normalize the search header
        search_header = section_header.strip()

        for section in sections:
            if section["header"].strip() == search_header:
                return section
            # Also try matching just the title
            if section["title"].lower() == search_header.lstrip("#").strip().lower():
                return section

        return None

    def replace_section(self, content: str, section_header: str, new_content: str) -> str | None:
        """Replace a section's content.

        Args:
            content: Full markdown content.
            section_header: Section header to replace.
            new_content: New content for the section (without header).

        Returns:
            Updated content or None if section not found.
        """
        post = frontmatter.loads(content)
        body = post.content

        section = self.find_section(content, section_header)
        if not section:
            return None

        # Build the new section
        new_section = f"{section['header']}\n\n{new_content.strip()}"

        # Replace in body
        new_body = body[: section["start"]] + new_section + "\n\n" + body[section["end"] :].lstrip()

        # Reconstruct with frontmatter
        post.content = new_body.strip()
        return frontmatter.dumps(post)

    def add_backlink(self, content: str, target: str, context: str | None = None) -> str:
        """Add a backlink to a document.

        Args:
            content: Full markdown content.
            target: Target note to link to.
            context: Optional context/reason for the link.

        Returns:
            Updated content with the backlink added.
        """
        post = frontmatter.loads(content)
        body = post.content

        # Check if there's a "Related" or similar section
        related_section = None
        for header in ["## Related", "## Related Notes", "## See Also", "## Links"]:
            related_section = self.find_section(content, header)
            if related_section:
                break

        link_text = f"- [[{target}]]"
        if context:
            link_text += f" - {context}"

        if related_section:
            # Add to existing section
            section_content = body[related_section["start"] : related_section["end"]]
            # Check if link already exists
            if f"[[{target}]]" in section_content:
                return content  # Already linked

            new_section_content = section_content.rstrip() + "\n" + link_text
            new_body = (
                body[: related_section["start"]]
                + new_section_content
                + "\n"
                + body[related_section["end"] :].lstrip()
            )
        else:
            # Create new Related section at the end
            new_body = body.rstrip() + f"\n\n## Related Notes\n\n{link_text}\n"

        post.content = new_body
        return frontmatter.dumps(post)

    def update_frontmatter(self, content: str, updates: dict) -> str:
        """Update frontmatter fields.

        Args:
            content: Full markdown content.
            updates: Dictionary of fields to update.

        Returns:
            Updated content.
        """
        post = frontmatter.loads(content)
        post.metadata.update(updates)
        return frontmatter.dumps(post)

    def scan_vault(self) -> list[Path]:
        """Scan the vault for all markdown files.

        Returns:
            List of markdown file paths.
        """
        return list(self.vault_path.rglob("*.md"))

    def resolve_link(self, link_target: str) -> Path | None:
        """Resolve a backlink target to an actual file path.

        Args:
            link_target: The target from a [[backlink]].

        Returns:
            Path to the file or None if not found.
        """
        # Try exact match first
        exact_path = self.vault_path / f"{link_target}.md"
        if exact_path.exists():
            return exact_path

        # Try case-insensitive search
        target_lower = link_target.lower()
        for md_file in self.vault_path.rglob("*.md"):
            if md_file.stem.lower() == target_lower:
                return md_file

        # Try with different separators
        for separator in ["-", "_", " "]:
            normalized = (
                link_target.replace(" ", separator).replace("-", separator).replace("_", separator)
            )
            for md_file in self.vault_path.rglob("*.md"):
                if (
                    md_file.stem.lower().replace("-", separator).replace("_", separator)
                    == normalized.lower()
                ):
                    return md_file

        return None
