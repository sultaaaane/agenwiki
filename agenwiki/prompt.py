"""
agenwiki.prompt
───────────────
Builds the system-prompt fragment that teaches any LLM agent
how to use the persistent wiki.  Import this and prepend the
result to your system prompt — the agent gets wiki-awareness
for free, regardless of framework.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .core import read_index, read_schema

# ─────────────────────────────────────────────
# Prompt fragment builder
# ─────────────────────────────────────────────

_WIKI_PROMPT = """\
═══════════════════════════════════════════════
PERSISTENT WIKI MEMORY
═══════════════════════════════════════════════

You have access to a persistent, compounding knowledge wiki stored at:
  {wiki_path}

This wiki is shared across ALL agents and ALL sessions.
It accumulates knowledge over time — treat it as your long-term memory.

TOOLS AVAILABLE
───────────────
{tool_list}

MANDATORY BEHAVIOURS
────────────────────
1. BEFORE answering any non-trivial question, call wiki_search to check
   if relevant knowledge already exists.

2. AFTER learning something new, completing a task, or generating a useful
   analysis, write it back to the wiki so future agents benefit.

3. When you find a contradiction between what the wiki says and new
   information, flag it in the relevant page rather than silently overwriting.

4. Always use [[Page Name]] wikilink syntax when referencing other pages.

5. Every page you write must have YAML frontmatter with: title, type,
   tags, created/updated dates.

WIKI STRUCTURE
──────────────
  wiki/index.md     → start here; master catalog of all pages
  wiki/log.md       → append-only event history
  wiki/entities/    → people, systems, organisations, products
  wiki/concepts/    → ideas, frameworks, theories
  wiki/sources/     → one summary per ingested document
  wiki/synthesis/   → cross-cutting analysis and comparisons

{index_section}
═══════════════════════════════════════════════
"""

_INDEX_SECTION = """\
CURRENT INDEX (top of wiki/index.md)
─────────────────────────────────────
{index_preview}
"""


def build_system_prompt(
    wiki_path: str | Path,
    tool_names: Optional[list[str]] = None,
    include_index: bool = True,
    index_preview_chars: int = 2000,
    prefix: str = "",
    suffix: str = "",
) -> str:
    """
    Build a system-prompt fragment that makes any LLM agent wiki-aware.

    Parameters
    ----------
    wiki_path : str | Path
        Absolute path to the wiki root on disk.
    tool_names : list[str] | None
        Names of the wiki tools available in this agent's tool set.
        Defaults to the full standard set.
    include_index : bool
        Whether to embed the current wiki index in the prompt.
        Useful for small wikis; turn off for very large ones.
    index_preview_chars : int
        How many characters of index.md to embed (default 2000).
    prefix : str
        Text to prepend before the wiki block (e.g. agent role description).
    suffix : str
        Text to append after the wiki block (e.g. task-specific instructions).

    Returns
    -------
    str
        Ready-to-use system prompt string.
    """
    if tool_names is None:
        tool_names = [
            "wiki_search(query)",
            "wiki_read(page)",
            "wiki_write(page, content)",
            "wiki_ingest(source_text, title)",
            "wiki_lint()",
            "wiki_list_pages(subdir?)",
        ]

    tool_list = "\n".join(f"  • {t}" for t in tool_names)

    index_section = ""
    if include_index:
        try:
            index_text = read_index(wiki_path)
            preview = index_text[:index_preview_chars]
            if len(index_text) > index_preview_chars:
                preview += (
                    "\n... (truncated — use wiki_read('index.md') for full index)"
                )
            index_section = _INDEX_SECTION.format(index_preview=preview)
        except Exception:
            pass

    wiki_block = _WIKI_PROMPT.format(
        wiki_path=str(wiki_path),
        tool_list=tool_list,
        index_section=index_section,
    )

    parts = [p for p in [prefix, wiki_block, suffix] if p]
    return "\n\n".join(parts)
