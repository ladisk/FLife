"""
Run the Python examples embedded in the README and the documentation, so that
they cannot silently break on upgrades (issue #9).

Every ``.. code-block:: python`` in each documentation file is extracted and
executed in order, sharing a single namespace per file (later snippets build on
earlier ones, just as a reader would run them top to bottom). Blocks that
require the interactive GUI cannot run on a headless CI and are skipped, but are
still syntax-checked.

The test only checks that the examples *run*; it does not assert numerical
results (the first example uses an unseeded random signal).
"""
import os
import re

import matplotlib
matplotlib.use("Agg")          # headless: no window, plt.show() is a no-op
import matplotlib.pyplot as plt

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Documentation files that contain runnable examples.
DOC_FILES = [
    "README.rst",
    os.path.join("docs", "source", "FLife.rst"),
]

# A block containing any of these needs the interactive GUI / a display and is
# skipped on a headless test runner (but still syntax-checked).
_GUI_MARKERS = ("'GUI'", '"GUI"', "SpectralData()", "pick_point", "set_mesh")


def _python_blocks(rst_path):
    """Return [(lineno, code), ...] for each '.. code-block:: python' block."""
    with open(rst_path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    blocks, i, n = [], 0, len(lines)
    directive = re.compile(r"^\s*\.\.\s+code-block::\s+python\s*$")
    while i < n:
        if directive.match(lines[i]):
            j = i + 1
            while j < n and lines[j].strip() == "":      # skip blank lines
                j += 1
            if j >= n:
                break
            indent = len(lines[j]) - len(lines[j].lstrip())
            body, first = [], j
            while j < n:
                if lines[j].strip() == "":
                    body.append("")
                    j += 1
                    continue
                if len(lines[j]) - len(lines[j].lstrip()) < indent:
                    break
                body.append(lines[j][indent:])
                j += 1
            blocks.append((first + 1, "\n".join(body).rstrip()))
            i = j
        else:
            i += 1
    return blocks


def _testable(code):
    return not any(marker in code for marker in _GUI_MARKERS)


@pytest.mark.parametrize("doc", DOC_FILES)
def test_doc_examples(doc):
    path = os.path.join(ROOT, doc)
    blocks = _python_blocks(path)
    assert blocks, f"no '.. code-block:: python' blocks found in {doc}"

    ns = {}
    plt.show = lambda *a, **k: None      # never block on a figure
    cwd = os.getcwd()
    os.chdir(ROOT)                       # so np.load('data/...') resolves
    try:
        for lineno, code in blocks:
            # Syntax-check every block, including interactive ones we cannot run.
            try:
                compiled = compile(code, f"{doc}:{lineno}", "exec")
            except SyntaxError as exc:
                raise AssertionError(
                    f"{doc} example starting at line {lineno} has a syntax error: "
                    f"{exc}\n--- block ---\n{code}\n-------------"
                ) from exc
            # Only execute the non-interactive (non-GUI) blocks.
            if not _testable(code):
                continue
            try:
                exec(compiled, ns)
            except Exception as exc:     # noqa: BLE001 - report which block broke
                raise AssertionError(
                    f"{doc} example starting at line {lineno} failed: "
                    f"{type(exc).__name__}: {exc}\n--- block ---\n{code}\n-------------"
                ) from exc
            finally:
                plt.close("all")
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    for _doc in DOC_FILES:
        test_doc_examples(_doc)
    print("Documentation examples ran OK")
