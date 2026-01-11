# Mini Docs MCP

[![CI](https://github.com/YOUR_USERNAME/mini-docs-mcp/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/mini-docs-mcp/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **Model Context Protocol (MCP)** server that transforms your local Markdown files into an intelligent, searchable knowledge base with semantic search and bidirectional linking.

## Features

- **Hybrid Search**: Combines semantic (vector) search with metadata filtering
- **Backlink Graph**: Automatic bidirectional link detection and navigation
- **Active Writing**: AI agents can create, update, and link documents
- **Real-time Sync**: Watchdog-based file monitoring for instant updates
- **Local-First**: All data stays on your machine (SQLite + LanceDB)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/mini-docs-mcp.git
cd mini-docs-mcp

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Running the Server

```bash
# Using uv
uv run python -m mini_docs_mcp.server

# Or directly
python -m mini_docs_mcp.server

# With custom vault path
VAULT_PATH=/path/to/your/vault python -m mini_docs_mcp.server
```

### Docker Deployment

The server supports SSE (Server-Sent Events) transport mode for HTTP access, perfect for Docker deployments:

1. **Create a `.env` file** (copy from `.env.example`):
   ```bash
   cp .env.example .env
   ```

2. **Configure your settings** in `.env`:
   ```bash
   # Transport mode: stdio or sse
   MCP_TRANSPORT=sse

   # SSE server configuration
   MCP_HOST=0.0.0.0
   MCP_PORT=8000
   ```

3. **Start the server**:
   ```bash
   docker-compose up -d
   ```

4. **Change port** (edit `.env` and restart):
   ```bash
   # In .env file
   MCP_PORT=9000

   # Restart container
   docker-compose down
   docker-compose up -d
   ```

The SSE server will be accessible at `http://localhost:8000` (or your configured port).

### Claude Desktop Integration

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "mini-docs": {
      "command": "python",
      "args": ["-m", "mini_docs_mcp.server"],
      "env": {
        "VAULT_PATH": "/path/to/your/vault"
      }
    }
  }
}
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `search_vault` | Semantic + metadata hybrid search |
| `read_note` | Read note with backlinks and related docs |
| `write_note` | Create or overwrite a note |
| `patch_note` | Update a specific section |
| `link_notes` | Create backlinks between notes |
| `list_notes` | List notes with filters |
| `find_similar` | Find semantically similar documents |
| `suggest_links` | Get link suggestions |
| `find_broken_links` | Detect broken backlinks |
| `get_note_graph` | Get graph context around a note |
| `sync_status` | Check/force vault synchronization |

## MCP Resources

| URI | Description |
|-----|-------------|
| `vault://graph/summary` | Knowledge base health report |
| `vault://note/recent` | Recently modified notes |
| `vault://tags/all` | All tags with counts |
| `vault://orphans` | Documents with no links |

## Document Format

Documents use standard Markdown with YAML frontmatter:

```markdown
---
id: my-document
title: My Document Title
type: concept
tags: [tag1, tag2]
status: draft
---

# My Document Title

Content here with [[backlinks]] to other documents.

## Related Notes

- [[another-document]]
```

### Document Types

- `concept` - Explanatory documents
- `task` - Action items and todos
- `ref` - Reference material
- `log` - Meeting notes, journals
- `idea` - Brainstorming and drafts

## Development

```bash
# Install dev dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=mini_docs_mcp

# Lint
uv run ruff check .

# Format
uv run ruff format .
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MCP Server (FastMCP)                  │
├─────────────────────────────────────────────────────────┤
│  Tools: search, read, write, patch, link, sync, ...     │
│  Resources: graph/summary, note/recent, tags/all, ...   │
├─────────────────────────────────────────────────────────┤
│                     Engine Layer                         │
│  SearchEngine │ ReadEngine │ WriteEngine │ LinkEngine   │
├─────────────────────────────────────────────────────────┤
│                    Storage Layer                         │
│  SQLite (metadata, edges) │ LanceDB (vectors)           │
├─────────────────────────────────────────────────────────┤
│                    File System                           │
│  Markdown Files │ Watchdog (real-time sync)             │
└─────────────────────────────────────────────────────────┘
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
