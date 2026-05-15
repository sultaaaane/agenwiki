"""
agenwiki.core
─────────────
Low-level file-system operations for the wiki.
No LLM dependency here — pure read/write/search.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from sentence_transformers import SentenceTransformer, util

    _EMBEDDER = None
except ImportError:
    SentenceTransformer = None
    _EMBEDDER = None

# ─────────────────────────────────────────────
# Wiki initialisation
# ─────────────────────────────────────────────

SCHEMA_TEMPLATE = """\
# Wiki Schema

## Directory layout
- raw/          → immutable source documents (never modify)
- wiki/         → LLM-generated markdown pages
  - index.md    → master catalog (auto-updated on every ingest)
  - log.md      → append-only event log
  - entities/   → people, systems, organisations, products
  - concepts/   → ideas, frameworks, theories
  - sources/    → one summary page per ingested source
  - synthesis/  → cross-cutting analysis, comparisons, open questions

## Page frontmatter
Every wiki page must start with YAML frontmatter:
```
---
title: <Human readable title>
type: entity | concept | source | synthesis
tags: [tag1, tag2]
created: YYYY-MM-DD
updated: YYYY-MM-DD
sources: [list of raw/ filenames this page was derived from]
---
```

## Wikilinks
Use [[Page Name]] syntax for internal links.
Always link to existing pages when mentioning them.

## Ingest workflow
1. Read the raw source
2. Write a summary page under wiki/sources/<slug>.md
3. Update or create relevant entity and concept pages
4. Append an entry to wiki/log.md
5. Update wiki/index.md

## Query workflow
1. Read wiki/index.md to find relevant pages
2. Read those pages
3. Synthesise an answer with [[Page]] citations
4. Optionally save the answer as a new synthesis/ page

## Lint checklist
- Pages with no inbound wikilinks → orphans
- Claims that contradict each other across pages → flag
- Concepts mentioned in text but missing their own page → gap
- Log entries with no corresponding wiki page → stale log
"""

INDEX_TEMPLATE = """\
---
title: Wiki Index
type: index
updated: {date}
---

# Wiki Index

> Auto-maintained. Do not edit by hand.

## Sources
<!-- sources -->

## Entities
<!-- entities -->

## Concepts
<!-- concepts -->

## Synthesis
<!-- synthesis -->
"""

LOG_TEMPLATE = """\
---
title: Event Log
type: log
---

# Event Log

> Append-only. Format: `## [YYYY-MM-DD] <operation> | <title>`

"""


def init_wiki(path: str | Path) -> Path:
    """
    Create a fresh wiki at *path*.
    Safe to call on an existing wiki — skips files that already exist.
    """
    root = Path(path)
    dirs = [
        root / "raw",
        root / "wiki" / "entities",
        root / "wiki" / "concepts",
        root / "wiki" / "sources",
        root / "wiki" / "synthesis",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    schema_file = root / "WIKI_SCHEMA.md"
    if not schema_file.exists():
        schema_file.write_text(SCHEMA_TEMPLATE)

    index_file = root / "wiki" / "index.md"
    if not index_file.exists():
        index_file.write_text(INDEX_TEMPLATE.format(date=_today()))

    log_file = root / "wiki" / "log.md"
    if not log_file.exists():
        log_file.write_text(LOG_TEMPLATE)

    return root


# ─────────────────────────────────────────────
# Read / write
# ─────────────────────────────────────────────


def read_page(wiki_root: str | Path, page: str) -> str:
    """
    Read a wiki page.  *page* can be:
      - a relative path like "concepts/attention.md"
      - a page slug like "attention"  (searched across all subdirs)
    Returns the full text or raises FileNotFoundError.
    """
    root = Path(wiki_root) / "wiki"
    candidate = root / page
    if candidate.exists():
        return candidate.read_text()

    # slug search
    slug = _slugify(page)
    for md in root.rglob("*.md"):
        if md.stem == slug or md.stem == page:
            return md.read_text()

    raise FileNotFoundError(f"Wiki page not found: {page!r}")


def write_page(wiki_root: str | Path, page: str, content: str) -> Path:
    """
    Write (create or overwrite) a wiki page.
    *page* should be a relative path like "concepts/attention.md".
    Returns the absolute path written.
    """
    root = Path(wiki_root) / "wiki"
    target = root / page
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)

    embedder = _get_embedder()
    if embedder is not None:
        emb = embedder.encode(content).tolist()
        emb_target = target.with_suffix(".emb")
        with open(emb_target, "w") as f:
            json.dump(emb, f)

    return target


def delete_page(wiki_root: str | Path, page: str) -> bool:
    """Delete a wiki page. Returns True if deleted, False if not found."""
    root = Path(wiki_root) / "wiki"
    target = root / page
    if target.exists():
        target.unlink()
        emb_target = target.with_suffix(".emb")
        if emb_target.exists():
            emb_target.unlink()
        return True
    return False


def list_pages(wiki_root: str | Path, subdir: Optional[str] = None) -> list[dict]:
    """
    Return a list of dicts with keys: path, title, type, tags, updated.
    *subdir* filters to a specific subdirectory (e.g. "concepts").
    """
    root = Path(wiki_root) / "wiki"
    search_root = root / subdir if subdir else root
    pages = []
    for md in sorted(search_root.rglob("*.md")):
        rel = md.relative_to(root)
        fm = _parse_frontmatter(md.read_text())
        pages.append(
            {
                "path": str(rel),
                "title": fm.get("title", md.stem),
                "type": fm.get("type", "unknown"),
                "tags": fm.get("tags", []),
                "updated": fm.get("updated", ""),
            }
        )
    return pages


# ─────────────────────────────────────────────
# Search
# ─────────────────────────────────────────────


def search_wiki(wiki_root: str | Path, query: str, top_k: int = 8) -> list[dict]:
    """
    Search over all wiki pages. Uses semantic search if embeddings are available,
    otherwise falls back to BM25-style keyword search.
    Returns ranked list of {path, title, snippet, score}.
    """
    root = Path(wiki_root) / "wiki"
    embedder = _get_embedder()

    if embedder is not None:
        from sentence_transformers import util

        query_emb = embedder.encode(query, convert_to_tensor=True)
        results = []
        for md in root.rglob("*.md"):
            if md.name in ("index.md", "log.md"):
                continue
            emb_file = md.with_suffix(".emb")
            if not emb_file.exists():
                continue
            try:
                with open(emb_file) as f:
                    doc_emb = json.load(f)
                sim = util.cos_sim(query_emb, doc_emb).item()
                text = md.read_text()
                fm = _parse_frontmatter(text)
                snippet = text[:200].replace("\n", " ").strip()
                results.append(
                    {
                        "path": str(md.relative_to(root)),
                        "title": fm.get("title", md.stem),
                        "score": sim,
                        "snippet": snippet,
                    }
                )
            except Exception:
                pass
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    terms = re.findall(r"\w+", query.lower())
    results = []

    for md in root.rglob("*.md"):
        if md.name in ("index.md", "log.md"):
            continue
        text = md.read_text()
        text_lower = text.lower()
        score = sum(text_lower.count(t) for t in terms)
        if score == 0:
            continue
        # extract a snippet around the first hit
        first_term = next((t for t in terms if t in text_lower), None)
        snippet = ""
        if first_term:
            idx = text_lower.find(first_term)
            start = max(0, idx - 80)
            end = min(len(text), idx + 200)
            snippet = text[start:end].replace("\n", " ").strip()

        fm = _parse_frontmatter(text)
        results.append(
            {
                "path": str(md.relative_to(root)),
                "title": fm.get("title", md.stem),
                "score": score,
                "snippet": snippet,
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# ─────────────────────────────────────────────
# Index & log helpers
# ─────────────────────────────────────────────


def append_log(
    wiki_root: str | Path, operation: str, title: str, notes: str = ""
) -> None:
    """Append one line to wiki/log.md."""
    log_path = Path(wiki_root) / "wiki" / "log.md"
    if not log_path.exists():
        log_path.write_text(LOG_TEMPLATE)
    entry = f"\n## [{_today()}] {operation} | {title}\n"
    if notes:
        entry += f"\n{notes}\n"
    with open(log_path, "a") as f:
        f.write(entry)


def rebuild_index(wiki_root: str | Path) -> None:
    """Regenerate wiki/index.md from the current set of pages."""
    root = Path(wiki_root) / "wiki"
    sections: dict[str, list[str]] = {
        "sources": [],
        "entities": [],
        "concepts": [],
        "synthesis": [],
    }
    for md in sorted(root.rglob("*.md")):
        rel = md.relative_to(root)
        if rel.parts[0] in ("index.md", "log.md") or md.name in ("index.md", "log.md"):
            continue
        fm = _parse_frontmatter(md.read_text())
        page_type = fm.get("type", "")
        title = fm.get("title", md.stem)
        link = f"- [[{md.stem}]] — {title}"
        bucket = page_type if page_type in sections else "synthesis"
        sections[bucket].append(link)

    content = INDEX_TEMPLATE.format(date=_today())
    for section, items in sections.items():
        block = "\n".join(items) if items else "_none yet_"
        content = content.replace(f"<!-- {section} -->", block)

    (root / "index.md").write_text(content)


def read_index(wiki_root: str | Path) -> str:
    index_path = Path(wiki_root) / "wiki" / "index.md"
    if not index_path.exists():
        rebuild_index(wiki_root)
    return index_path.read_text()


def read_schema(wiki_root: str | Path) -> str:
    schema_path = Path(wiki_root) / "WIKI_SCHEMA.md"
    if not schema_path.exists():
        init_wiki(wiki_root)
    return schema_path.read_text()


# ─────────────────────────────────────────────
# Raw source helpers
# ─────────────────────────────────────────────


def add_raw_source(wiki_root: str | Path, filename: str, content: str) -> Path:
    """Save a document to raw/ for later ingestion."""
    raw_dir = Path(wiki_root) / "raw"
    raw_dir.mkdir(exist_ok=True)
    target = raw_dir / filename
    target.write_text(content)
    return target


def list_raw_sources(wiki_root: str | Path) -> list[str]:
    raw_dir = Path(wiki_root) / "raw"
    if not raw_dir.exists():
        return []
    return [f.name for f in sorted(raw_dir.iterdir()) if f.is_file()]


def read_raw_source(wiki_root: str | Path, filename: str) -> str:
    return (Path(wiki_root) / "raw" / filename).read_text()


# ─────────────────────────────────────────────
# Lint
# ─────────────────────────────────────────────


def lint_wiki(wiki_root: str | Path) -> dict:
    """
    Health-check the wiki.
    Returns dict with: orphans, broken_links, missing_pages.
    """
    root = Path(wiki_root) / "wiki"
    all_pages: dict[str, str] = {}  # stem -> path
    for md in root.rglob("*.md"):
        all_pages[md.stem] = str(md.relative_to(root))

    inbound: dict[str, int] = {stem: 0 for stem in all_pages}
    broken: list[dict] = []

    for md in root.rglob("*.md"):
        text = md.read_text()
        links = re.findall(r"\[\[([^\]]+)\]\]", text)
        for link in links:
            slug = _slugify(link)
            if slug in all_pages:
                inbound[slug] = inbound.get(slug, 0) + 1
            elif link in all_pages:
                inbound[link] = inbound.get(link, 0) + 1
            else:
                broken.append({"from": str(md.relative_to(root)), "link": link})

    skip = {"index", "log"}
    orphans = [
        {"path": path, "stem": stem}
        for stem, path in all_pages.items()
        if inbound.get(stem, 0) == 0 and stem not in skip
    ]

    return {
        "orphans": orphans,
        "broken_links": broken,
        "total_pages": len(all_pages),
        "total_orphans": len(orphans),
        "total_broken": len(broken),
    }


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _parse_frontmatter(text: str) -> dict:
    """Very small YAML frontmatter parser (no deps)."""
    if not text.startswith("---"):
        return {}
    end = text.find("---", 3)
    if end == -1:
        return {}
    fm_text = text[3:end].strip()
    result: dict = {}
    for line in fm_text.splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        # handle lists like: tags: [a, b, c]
        if val.startswith("[") and val.endswith("]"):
            val = [v.strip().strip("\"'") for v in val[1:-1].split(",") if v.strip()]
        result[key] = val
    return result


def _get_embedder():
    global _EMBEDDER
    if SentenceTransformer is not None and _EMBEDDER is None:
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return _EMBEDDER


def git_commit(wiki_root: str | Path, message: str = "wiki update") -> None:
    """Commit all changes in the wiki directory to git."""
    root = str(wiki_root)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-m", message], cwd=root, check=True)


def git_diff(wiki_root: str | Path) -> str:
    """Return the git diff for the last commit."""
    root = str(wiki_root)
    result = subprocess.run(
        ["git", "diff", "HEAD~1"], cwd=root, capture_output=True, text=True
    )
    return result.stdout
