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
