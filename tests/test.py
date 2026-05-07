from agenwiki.core import init_wiki
import tempfile, pathlib

with tempfile.TemporaryDirectory() as tmp:
    root = init_wiki(tmp)
    assert (pathlib.Path(tmp) / "wiki" / "index.md").exists()
    init_wiki(tmp)
    print("step 2 ok")
