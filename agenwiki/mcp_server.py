"""
agenwiki.mcp_server
───────────────────
Expose the wiki as an MCP (Model Context Protocol) server.

Run with:
    python -m agenwiki.mcp_server --wiki ./my-wiki

Then add to Claude Desktop / Claude Code config:
    {
      "mcpServers": {
        "wiki": {
          "command": "python",
          "args": ["-m", "agenwiki.mcp_server", "--wiki", "/abs/path/to/my-wiki"]
        }
      }
    }

Requires: pip install mcp
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import core
from .tools import make_python_tools


def run_server(wiki_path: str | Path) -> None:
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp import types
    except ImportError:
        print(
            "ERROR: 'mcp' package not installed.\n" "Run: pip install mcp",
            file=sys.stderr,
        )
        sys.exit(1)

    root = Path(wiki_path).resolve()
    core.init_wiki(root)
    py_tools = make_python_tools(root)
    server = Server("agenwiki")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="wiki_search",
                description="Search the persistent wiki for pages matching a query.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "default": 8},
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="wiki_read",
                description="Read a wiki page by path or slug.",
                inputSchema={
                    "type": "object",
                    "properties": {"page": {"type": "string"}},
                    "required": ["page"],
                },
            ),
            types.Tool(
                name="wiki_write",
                description="Write or update a wiki page (include YAML frontmatter).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "page": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["page", "content"],
                },
            ),
            types.Tool(
                name="wiki_ingest",
                description="Save a source document and create a summary stub.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source_text": {"type": "string"},
                        "title": {"type": "string"},
                        "source_filename": {"type": "string"},
                    },
                    "required": ["source_text", "title"],
                },
            ),
            types.Tool(
                name="wiki_lint",
                description="Health-check the wiki for orphans and broken links.",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="wiki_list_pages",
                description="List all wiki pages with metadata.",
                inputSchema={
                    "type": "object",
                    "properties": {"subdir": {"type": "string"}},
                },
            ),
            types.Tool(
                name="wiki_read_index",
                description="Read the master wiki index.",
                inputSchema={"type": "object", "properties": {}},
            ),
            types.Tool(
                name="wiki_append_log",
                description="Append an event to the wiki log.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string"},
                        "title": {"type": "string"},
                        "notes": {"type": "string"},
                    },
                    "required": ["operation", "title"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        if name not in py_tools:
            result = f"ERROR: unknown tool {name!r}"
        else:
            try:
                result = py_tools[name](**arguments)
            except Exception as exc:
                result = f"ERROR: {name} failed — {exc}"
        return [types.TextContent(type="text", text=str(result))]

    import asyncio

    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream, write_stream, server.create_initialization_options()
            )

    asyncio.run(main())


def main():
    parser = argparse.ArgumentParser(description="agenwiki MCP server")
    parser.add_argument("--wiki", required=True, help="Path to wiki root directory")
    args = parser.parse_args()
    run_server(args.wiki)


if __name__ == "__main__":
    main()
