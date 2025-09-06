import doctest
from pathlib import Path


def _docs_with_pycon_blocks(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*.md"):
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        if "```pycon" in text:
            out.append(p)
    return out


def test_docs_doctest_pycon_blocks():
    docs = _docs_with_pycon_blocks(Path("docs"))
    for doc in docs:
        result = doctest.testfile(
            str(doc), module_relative=False, optionflags=doctest.ELLIPSIS
        )
        assert (
            result.failed == 0
        ), f"Doctest failed in {doc}: {result.failed} failures out of {result.attempted} checks"
