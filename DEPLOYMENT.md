# Deployment Guide

This document provides detailed deployment instructions for Mini Docs MCP server.

## FastMCP Cloud Deployment

[FastMCP Cloud](https://fastmcp.cloud) is a free managed hosting platform for MCP servers (currently in beta).

### Prerequisites

- GitHub account
- This repository pushed to GitHub
- `pyproject.toml` with dependencies (included)
- `fastmcp.json` configuration file (included)

### Step-by-Step Instructions

1. **Visit FastMCP Cloud**

   Go to [fastmcp.cloud](https://fastmcp.cloud) and sign in with your GitHub account.

2. **Create a New Project**

   - Click "Create Project"
   - Select this repository from your GitHub account
   - Alternatively, you can fork the FastMCP Cloud quickstart repo

3. **Configure Project Settings**

   FastMCP Cloud will auto-detect most settings from `fastmcp.json`, but you can customize:

   - **Project Name**: `mini-docs-mcp` (creates unique URL: `https://mini-docs-mcp-yourorg.fastmcp.cloud`)
   - **Entrypoint**: `src/mini_docs_mcp/server.py:mcp` (auto-detected)
   - **Python Version**: 3.11+ (auto-detected from `pyproject.toml`)
   - **Authentication**:
     - Disabled: Public access to anyone with the URL
     - Enabled: Only your FastMCP Cloud organization members can access

4. **Environment Variables** (Optional)

   Configure these in the FastMCP Cloud dashboard:

   ```bash
   VAULT_PATH=/data/vault        # Path to your vault directory
   MCP_TRANSPORT=sse             # Transport mode (SSE for HTTP)
   MCP_HOST=0.0.0.0              # Server host
   MCP_PORT=8000                 # Server port
   ```

   Default values from `fastmcp.json` will be used if not specified.

5. **Deploy**

   - Click "Deploy"
   - FastMCP Cloud will:
     - Install dependencies from `pyproject.toml`
     - Set up the Python environment with `uv`
     - Start your MCP server
     - Provide a unique URL for your server

6. **Access Your Server**

   Once deployed, you'll receive a URL like:
   ```
   https://mini-docs-mcp-yourorg.fastmcp.cloud
   ```

   The MCP endpoint will be available at:
   ```
   https://mini-docs-mcp-yourorg.fastmcp.cloud/mcp
   ```

### Automatic Deployments

FastMCP Cloud includes built-in CI/CD:

- **Pull Requests**: Opening a PR automatically creates a preview deployment
- **Main Branch**: Pushing to main automatically updates your production deployment
- **Branch Deployments**: Each PR gets its own isolated environment

### Upcoming CLI Support

Once the `fastmcp deploy` command is available, deployment will be even simpler:

```bash
# Deploy directly from terminal
fastmcp deploy

# Deploy with custom configuration
fastmcp deploy --config fastmcp.json
```

## Docker Deployment

For self-hosted deployments, use Docker Compose.

### Prerequisites

- Docker and Docker Compose installed
- `.env` file (copy from `.env.example`)

### Quick Start

1. **Create `.env` file**:
   ```bash
   cp .env.example .env
   ```

2. **Configure settings** in `.env`:
   ```bash
   MCP_TRANSPORT=sse
   MCP_HOST=0.0.0.0
   MCP_PORT=8000
   VAULT_PATH=/data/vault
   ```

3. **Start the server**:
   ```bash
   docker-compose up -d
   ```

4. **Access the server**:
   ```
   http://localhost:8000
   ```

### Docker Configuration

The server uses:
- Python 3.11 base image
- `uv` for fast dependency installation
- Volume mounts for vault and data persistence
- SSE transport for HTTP access

## Local Development

For local development without Docker:

```bash
# Install dependencies
uv sync

# Run server (stdio mode)
uv run python -m mini_docs_mcp.server

# Run server (SSE mode)
MCP_TRANSPORT=sse MCP_PORT=8000 uv run python -m mini_docs_mcp.server

# With custom vault path
VAULT_PATH=/path/to/vault uv run python -m mini_docs_mcp.server
```

## Configuration Reference

### `fastmcp.json`

```json
{
  "name": "mini-docs-mcp",
  "version": "0.1.0",
  "description": "MCP server for intelligent Markdown knowledge base",
  "entrypoint": "src/mini_docs_mcp/server.py:mcp",
  "python": "3.11",
  "environment": {
    "VAULT_PATH": "/data/vault",
    "MCP_TRANSPORT": "sse",
    "MCP_HOST": "0.0.0.0",
    "MCP_PORT": "8000"
  },
  "deployment": {
    "transport": "sse",
    "port": 8000
  }
}
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VAULT_PATH` | Path to Markdown vault directory | `./vault` |
| `MCP_TRANSPORT` | Transport mode: `stdio` or `sse` | `stdio` |
| `MCP_HOST` | Server host (SSE mode only) | `0.0.0.0` |
| `MCP_PORT` | Server port (SSE mode only) | `8000` |

## Troubleshooting

### FastMCP Cloud Issues

- **Dependencies not installing**: Ensure `pyproject.toml` has all required dependencies
- **Server not starting**: Check logs in FastMCP Cloud dashboard
- **Authentication errors**: Verify authentication settings in project configuration

### Docker Issues

- **Port already in use**: Change `MCP_PORT` in `.env` file
- **Permission errors**: Ensure vault directory has correct permissions
- **Container not starting**: Check logs with `docker-compose logs`

### General Issues

- **Embedding model loading**: First startup takes longer as it downloads the sentence-transformers model
- **Memory usage**: Vector embeddings require ~500MB RAM minimum
- **File watcher**: May need increased inotify limits on Linux

## Support

For issues and questions:
- GitHub Issues: [mini-docs-mcp/issues](https://github.com/YOUR_USERNAME/mini-docs-mcp/issues)
- FastMCP Cloud: [fastmcp.cloud/support](https://fastmcp.cloud/support)
- FastMCP Docs: [gofastmcp.com](https://gofastmcp.com)
