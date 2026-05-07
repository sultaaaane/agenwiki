"""
agenwiki.tools
──────────────
Framework-specific tool wrappers around agenwiki.core.

Import only what you need:

    from agenwiki.tools import make_python_tools          # plain callables
    from agenwiki.tools import make_langchain_tools       # LangChain / LangGraph
    from agenwiki.tools import make_claude_tool_schemas   # Anthropic API JSON
    from agenwiki.tools import make_openai_tool_schemas   # OpenAI / LiteLLM JSON
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from . import core

# ─────────────────────────────────────────────
# 1. Plain Python callables
#    Works with: raw LLM loops, AutoGen, CrewAI,
#                any framework that accepts plain functions
# ─────────────────────────────────────────────


def make_python_tools(wiki_path: str | Path) -> dict[str, Callable]:
    """
    Returns a dict of {name: callable} for all wiki operations.
    The callables are fully bound to *wiki_path* — no path argument needed
    when calling them.

    Usage
    -----
        tools = make_python_tools("./wiki")
        result = tools["wiki_search"]("attention mechanism")
        tools["wiki_write"]("concepts/attention.md", page_content)
    """
    root = Path(wiki_path)

    def wiki_search(query: str, top_k: int = 8) -> str:
        """Search the wiki for pages matching *query*. Returns JSON."""
        results = core.search_wiki(root, query, top_k=top_k)
        if not results:
            return json.dumps({"results": [], "message": "No matching pages found."})
        return json.dumps({"results": results})

    def wiki_read(page: str) -> str:
        """Read a wiki page by relative path or slug."""
        try:
            return core.read_page(root, page)
        except FileNotFoundError:
            return f"ERROR: Page not found: {page!r}"

    def wiki_write(page: str, content: str) -> str:
        """Write (create or update) a wiki page. Returns confirmation."""
        path = core.write_page(root, page, content)
        core.rebuild_index(root)
        return f"OK: wrote {path}"

    def wiki_ingest(
        source_text: str, title: str, source_filename: Optional[str] = None
    ) -> str:
        """
        Save *source_text* as a raw document and create a stub sources/ page.
        The LLM should then enrich the stub by calling wiki_write.
        """
        slug = core._slugify(title)
        fname = source_filename or f"{slug}.md"
        core.add_raw_source(root, fname, source_text)
        # Create a stub page the agent will fill in
        today = core._today()
        stub = f"""\
---
title: {title}
type: source
tags: []
created: {today}
updated: {today}
sources: [{fname}]
---
 
# {title}
 
> **TODO**: Summarise this source, extract key entities and concepts,
> and update relevant wiki pages.
 
## Raw source reference
See `raw/{fname}`.
"""
        stub_path = f"sources/{slug}.md"
        core.write_page(root, stub_path, stub)
        core.append_log(root, "ingest", title, f"stub created at wiki/{stub_path}")
        core.rebuild_index(root)
        return f"OK: source saved to raw/{fname}, stub at wiki/{stub_path}"

    def wiki_lint() -> str:
        """Run a health-check on the wiki. Returns JSON with orphans and broken links."""
        return json.dumps(core.lint_wiki(root), indent=2)

    def wiki_list_pages(subdir: Optional[str] = None) -> str:
        """List all wiki pages, optionally filtered to a subdirectory."""
        pages = core.list_pages(root, subdir)
        return json.dumps({"pages": pages})

    def wiki_read_index() -> str:
        """Read the master index page."""
        return core.read_index(root)

    def wiki_append_log(operation: str, title: str, notes: str = "") -> str:
        """Append an entry to the event log."""
        core.append_log(root, operation, title, notes)
        return f"OK: logged [{operation}] {title}"

    return {
        "wiki_search": wiki_search,
        "wiki_read": wiki_read,
        "wiki_write": wiki_write,
        "wiki_ingest": wiki_ingest,
        "wiki_lint": wiki_lint,
        "wiki_list_pages": wiki_list_pages,
        "wiki_read_index": wiki_read_index,
        "wiki_append_log": wiki_append_log,
    }


# ─────────────────────────────────────────────
# 2. LangChain / LangGraph tools
# ─────────────────────────────────────────────


def make_langchain_tools(wiki_path: str | Path) -> list:
    """
    Returns a list of LangChain StructuredTool instances.
    Drop them into create_react_agent, AgentExecutor, or any LC graph.

    Usage
    -----
        from agenwiki.tools import make_langchain_tools
        tools = make_langchain_tools("./my-wiki")
        agent = create_react_agent(llm, tools)
    """
    try:
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field
    except ImportError as e:
        raise ImportError(
            "langchain-core and pydantic are required: pip install langchain-core pydantic"
        ) from e

    py_tools = make_python_tools(wiki_path)

    class SearchInput(BaseModel):
        query: str = Field(description="Search query string")
        top_k: int = Field(default=8, description="Number of results to return")

    class ReadInput(BaseModel):
        page: str = Field(
            description="Page path (e.g. 'concepts/attention.md') or slug"
        )

    class WriteInput(BaseModel):
        page: str = Field(description="Relative path like 'concepts/attention.md'")
        content: str = Field(description="Full markdown content including frontmatter")

    class IngestInput(BaseModel):
        source_text: str = Field(description="Text content of the source document")
        title: str = Field(description="Human-readable title for the source")
        source_filename: Optional[str] = Field(
            default=None, description="Optional filename"
        )

    class SubdirInput(BaseModel):
        subdir: Optional[str] = Field(
            default=None, description="Subdirectory to filter by"
        )

    class LogInput(BaseModel):
        operation: str = Field(
            description="Operation type, e.g. 'ingest', 'query', 'lint'"
        )
        title: str = Field(description="Short title of the event")
        notes: str = Field(default="", description="Optional notes")

    lc_tools = [
        StructuredTool.from_function(
            func=py_tools["wiki_search"],
            name="wiki_search",
            description="Search the persistent wiki for pages matching a query. Always call this before answering non-trivial questions.",
            args_schema=SearchInput,
        ),
        StructuredTool.from_function(
            func=py_tools["wiki_read"],
            name="wiki_read",
            description="Read the full content of a wiki page by path or slug.",
            args_schema=ReadInput,
        ),
        StructuredTool.from_function(
            func=py_tools["wiki_write"],
            name="wiki_write",
            description="Create or update a wiki page. Include YAML frontmatter. Rebuilds the index automatically.",
            args_schema=WriteInput,
        ),
        StructuredTool.from_function(
            func=py_tools["wiki_ingest"],
            name="wiki_ingest",
            description="Save a new source document and create a wiki stub for it.",
            args_schema=IngestInput,
        ),
        StructuredTool.from_function(
            func=py_tools["wiki_lint"],
            name="wiki_lint",
            description="Health-check the wiki: find orphan pages, broken links, and gaps.",
        ),
        StructuredTool.from_function(
            func=py_tools["wiki_list_pages"],
            name="wiki_list_pages",
            description="List all wiki pages with metadata.",
            args_schema=SubdirInput,
        ),
        StructuredTool.from_function(
            func=py_tools["wiki_read_index"],
            name="wiki_read_index",
            description="Read the master wiki index to orient yourself.",
        ),
        StructuredTool.from_function(
            func=py_tools["wiki_append_log"],
            name="wiki_append_log",
            description="Append an event to the wiki log.",
            args_schema=LogInput,
        ),
    ]
    return lc_tools


# ─────────────────────────────────────────────
# 3. Anthropic (Claude) API tool schemas
#    Use with client.messages.create(tools=...)
# ─────────────────────────────────────────────


def make_claude_tool_schemas() -> list[dict]:
    """
    Returns tool definitions in Anthropic's JSON schema format.
    Pair with the callables from make_python_tools() to handle tool_use blocks.

    Usage
    -----
        schemas = make_claude_tool_schemas()
        py_tools = make_python_tools("./my-wiki")

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            tools=schemas,
            messages=[...]
        )

        # handle tool_use blocks:
        for block in response.content:
            if block.type == "tool_use":
                result = py_tools[block.name](**block.input)
    """
    return [
        {
            "name": "wiki_search",
            "description": "Search the persistent wiki for relevant pages. Call this before answering non-trivial questions.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {
                        "type": "integer",
                        "description": "Max results",
                        "default": 8,
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "wiki_read",
            "description": "Read the full content of a wiki page.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "page": {"type": "string", "description": "Page path or slug"},
                },
                "required": ["page"],
            },
        },
        {
            "name": "wiki_write",
            "description": "Write or update a wiki page with markdown content (include YAML frontmatter).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "page": {
                        "type": "string",
                        "description": "Relative path, e.g. 'concepts/attention.md'",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full markdown content",
                    },
                },
                "required": ["page", "content"],
            },
        },
        {
            "name": "wiki_ingest",
            "description": "Save a new source document to the wiki and create a summary stub.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "source_text": {
                        "type": "string",
                        "description": "Content of the source",
                    },
                    "title": {"type": "string", "description": "Title for the source"},
                    "source_filename": {
                        "type": "string",
                        "description": "Optional filename",
                    },
                },
                "required": ["source_text", "title"],
            },
        },
        {
            "name": "wiki_lint",
            "description": "Health-check the wiki for orphan pages and broken links.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "wiki_list_pages",
            "description": "List all wiki pages with metadata.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "subdir": {
                        "type": "string",
                        "description": "Optional subdirectory filter",
                    },
                },
            },
        },
        {
            "name": "wiki_read_index",
            "description": "Read the master index to find relevant pages.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "wiki_append_log",
            "description": "Append an event to the wiki log.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {"type": "string"},
                    "title": {"type": "string"},
                    "notes": {"type": "string"},
                },
                "required": ["operation", "title"],
            },
        },
    ]


# ─────────────────────────────────────────────
# 4. OpenAI / LiteLLM tool schemas
# ─────────────────────────────────────────────


def make_openai_tool_schemas() -> list[dict]:
    """
    Returns tool definitions in OpenAI's function-calling format.
    Works with LiteLLM, Azure OpenAI, and any OpenAI-compatible API.
    """
    claude_schemas = make_claude_tool_schemas()
    openai_schemas = []
    for tool in claude_schemas:
        openai_schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
        )
    return openai_schemas


# ─────────────────────────────────────────────
# 5. Generic tool-call dispatcher
#    Works with any framework: receive a tool name + input dict,
#    execute it, return a string result.
# ─────────────────────────────────────────────


def make_dispatcher(wiki_path: str | Path) -> Callable[[str, dict], str]:
    """
    Returns a single dispatch function: (tool_name, input_dict) -> str result.
    Use this in your agent loop to handle tool calls uniformly.

    Usage
    -----
        dispatch = make_dispatcher("./my-wiki")

        # in your agent loop:
        for tool_call in response.tool_calls:
            result = dispatch(tool_call.name, tool_call.input)
            # feed result back to the model
    """
    py_tools = make_python_tools(wiki_path)

    def dispatch(tool_name: str, tool_input: dict) -> str:
        if tool_name not in py_tools:
            return f"ERROR: unknown tool {tool_name!r}. Available: {list(py_tools)}"
        try:
            result = py_tools[tool_name](**tool_input)
            return str(result)
        except Exception as exc:
            return f"ERROR: {tool_name} failed — {exc}"

    return dispatch
