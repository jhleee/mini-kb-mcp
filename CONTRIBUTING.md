# Contributing to Mini Docs MCP

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/mini-docs-mcp.git
   cd mini-docs-mcp
   ```

2. **Install uv** (recommended package manager)
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. **Install dependencies**
   ```bash
   uv sync --all-extras
   ```

4. **Run tests**
   ```bash
   uv run pytest
   ```

## Code Style

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting.

```bash
# Check linting
uv run ruff check .

# Auto-fix issues
uv run ruff check --fix .

# Format code
uv run ruff format .
```

### Style Guidelines

- Use type hints for all function parameters and return values
- Write docstrings in Google style
- Keep functions focused and single-purpose
- Add tests for new functionality

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=mini_docs_mcp

# Run specific test file
uv run pytest tests/test_parser.py

# Run with verbose output
uv run pytest -v
```

### Test Categories

- `tests/test_parser.py` - Markdown parsing tests
- `tests/test_sqlite.py` - Database operation tests
- `tests/test_integration.py` - Integration tests
- `tests/test_hardcore.py` - Stress and edge case tests

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** for your feature/fix
   ```bash
   git checkout -b feature/my-feature
   ```
3. **Make your changes** with appropriate tests
4. **Run the test suite** to ensure nothing is broken
   ```bash
   uv run pytest
   uv run ruff check .
   ```
5. **Commit** with a clear message
   ```bash
   git commit -m "Add feature: description of change"
   ```
6. **Push** and create a Pull Request

## Commit Messages

Follow conventional commit format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring
- `chore:` Maintenance tasks

Example:
```
feat: add tag filtering to search_vault tool

- Added tag_filter parameter
- Updated search engine to handle multiple tags
- Added tests for tag filtering
```

## Reporting Issues

When reporting bugs, please include:

1. Python version (`python --version`)
2. Operating system
3. Steps to reproduce
4. Expected vs actual behavior
5. Error messages/stack traces

## Feature Requests

For feature requests:

1. Check existing issues first
2. Describe the use case
3. Explain the expected behavior
4. Consider if it fits the project scope

## Questions?

Feel free to open an issue for questions or discussions.
