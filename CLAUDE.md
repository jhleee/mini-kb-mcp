# Mini Docs MCP - Development Guide

## Project Overview

MCP server for intelligent Markdown knowledge base with semantic search and bidirectional linking.

## Tech Stack

- **Python**: 3.11+
- **MCP Framework**: FastMCP
- **Vector DB**: LanceDB
- **Metadata DB**: SQLite
- **Embedding**: Sentence-Transformers (all-MiniLM-L6-v2)
- **File Watcher**: Watchdog

## Commands

```bash
# Install
uv sync

# Run server (stdio)
uv run python -m mini_docs_mcp.server

# Run server (SSE)
MCP_TRANSPORT=sse MCP_PORT=8000 uv run python -m mini_docs_mcp.server

# Test
uv run pytest

# Lint
uv run ruff check .

# Format
uv run ruff format .
```

## Project Structure

```
src/mini_docs_mcp/
├── server.py          # MCP server entry point
├── db/
│   ├── sqlite.py      # Metadata storage
│   └── vector.py      # Vector embeddings
├── parser/
│   └── markdown.py    # Frontmatter & backlink parser
├── tools/
│   ├── search.py      # search_vault
│   ├── read.py        # read_note
│   ├── write.py       # write_note, patch_note
│   └── link.py        # link_notes
├── resources/
│   └── vault.py       # vault:// resources
└── watcher/
    └── sync.py        # Real-time file sync
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `search_vault` | Hybrid semantic + metadata search |
| `read_note` | Read note with backlinks |
| `write_note` | Create/overwrite note |
| `patch_note` | Update specific section |
| `link_notes` | Add backlink between notes |
| `list_notes` | List with filters |
| `find_similar` | Find similar documents |
| `suggest_links` | Get link suggestions |
| `find_broken_links` | Detect broken links |
| `get_note_graph` | Graph context (2-hop) |
| `sync_status` | Force sync |

## Deployment

### FastMCP Cloud

This project is optimized for deployment to [FastMCP Cloud](https://fastmcp.cloud):

- **Configuration**: `fastmcp.json` in project root
- **Entrypoint**: `src/mini_docs_mcp/server.py:mcp`
- **Dependencies**: Auto-detected from `pyproject.toml`
- **Transport**: SSE (Server-Sent Events) for HTTP access

### Docker

Use `docker-compose.yaml` for local Docker deployment:
- SSE server on configurable port (default: 8000)
- Environment variables in `.env` file

## Coding Conventions

- Type hints required
- Google-style docstrings
- Async/await patterns
- `logging` module for logs
