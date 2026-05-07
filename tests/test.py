from agenwiki.core import *
from agenwiki.tools import *
from agenwiki.prompt import *
import tempfile, pathlib

with tempfile.TemporaryDirectory() as tmp:
    root = init_wiki(tmp)
    assert (pathlib.Path(tmp) / "wiki" / "index.md").exists()
    init_wiki(tmp)
    print("step 2 ok")

content = "---\ntitle: Attention\ntype: concept\n---\n# Attention\nSelf-attention."
write_page(root, "concepts/attention.md", content)
assert read_page(root, "concepts/attention.md") == content
assert read_page(root, "attention") == content  # slug lookup
try:
    read_page(root, "missing")
    assert False
except FileNotFoundError:
    print("step 3 ok")

write_page(root, "concepts/attention.md", "---\ntitle: Attention\ntype: concept\n---")
write_page(root, "entities/vaswani.md", "---\ntitle: Vaswani\ntype: entity\n---")
rebuild_index(root)
index = read_index(root)
assert "attention" in index.lower()
assert "vaswani" in index.lower()
print("step 4 ok")

write_page(
    root,
    "concepts/transformers.md",
    "---\ntitle: Transformers\ntype: concept\n---\nSelf-attention mechanism.",
)
write_page(
    root,
    "concepts/rnn.md",
    "---\ntitle: RNN\ntype: concept\n---\nRecurrent neural network.",
)

results = search_wiki(root, "attention")
assert results[0]["path"] == "concepts/transformers.md"
assert search_wiki(root, "zzznomatch") == []
print("step 5 ok")

append_log(root, "ingest", "Test Paper", "some notes")
log = (root / "wiki" / "log.md").read_text()
assert "Test Paper" in log

add_raw_source(root, "paper.txt", "content here")
assert "paper.txt" in list_raw_sources(root)
print("step 6 ok")

write_page(
    root,
    "concepts/orphan.md",
    "---\ntitle: Orphan\ntype: concept\n---\nNo one links here.",
)
write_page(
    root,
    "concepts/broken.md",
    "---\ntitle: Broken\ntype: concept\n---\nSee [[GhostPage]] for details.",
)

result = lint_wiki(root)
assert any(o["stem"] == "orphan" for o in result["orphans"])
assert any(b["link"] == "GhostPage" for b in result["broken_links"])
print("step 7 ok")

tools = make_python_tools(root)

tools["wiki_write"](
    "concepts/attn.md", "---\ntitle: Attn\ntype: concept\n---\nAttention."
)
assert "Attn" in tools["wiki_read"]("attn")

import json

results = json.loads(tools["wiki_search"]("attention"))
assert len(results["results"]) >= 1

result = tools["wiki_ingest"]("Paper content.", "My Paper")
assert "OK" in result
print("step 8 ok")

dispatch = make_dispatcher(root)
result = dispatch("wiki_search", {"query": "attention"})
assert "results" in result

error = dispatch("nonexistent_tool", {})
assert "ERROR" in error
print("step 9 ok")

prompt = build_system_prompt(root, suffix="You are a pirate.")
assert "PERSISTENT WIKI MEMORY" in prompt
assert str(root) in prompt
assert "You are a pirate." in prompt
print("step 10 ok")
