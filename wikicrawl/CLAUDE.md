# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a web crawling project using the crawl4ai library with Fire for CLI parameter handling.

## Development Environment

**CRITICAL**: ALWAYS use `uv` for dependency management. NEVER edit `pyproject.toml` directly to add dependencies - use `uv add <package>` instead.

## Commands

### Dependency Management
```bash
# Add a new dependency
uv add <package-name>

# Install all dependencies
uv sync

# Update dependencies
uv lock --upgrade
```

### Running the Crawler
```bash
# Basic usage with default parameters
python main.py

# Specify depth and URL
python main.py --url "https://example.com" --depth 3

# Save results to JSON
python main.py --output results.json

# All parameters
python main.py --url "https://example.com" --depth 5 --include-external --output data/crawl.json --no-verbose
```

## Architecture

The project consists of a single `main.py` file that:
- Uses `fire` for CLI parameter parsing
- Implements async web crawling using `crawl4ai`
- Supports configurable crawl depth via BFS strategy
- Can save results to JSON files with metadata including URL, depth, title, description, content length, and link counts

## Key Parameters

- `url`: Target URL to crawl (default: "https://example.com")
- `depth`: Maximum crawl depth (default: 2)
- `include_external`: Whether to follow external links (default: False)
- `verbose`: Enable verbose output (default: True)
- `output`: Optional path to save JSON results