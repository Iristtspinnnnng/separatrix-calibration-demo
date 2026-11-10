"""Where things live.

Every script writes its outputs relative to the repository root, not the
current working directory, so ``python experiments/run_demo.py`` behaves the
same from anywhere::

    figures/   publication figures (committed)
    results/   CSV / JSON measurement tables (committed)
    .cache/    regenerable label rasters (git-ignored)
"""

from pathlib import Path

__all__ = ["REPO_ROOT", "FIGURES", "RESULTS", "CACHE", "DOCS_DATA",
           "figure", "result", "cache"]

REPO_ROOT = Path(__file__).resolve().parents[1]
FIGURES = REPO_ROOT / "figures"
RESULTS = REPO_ROOT / "results"
CACHE = REPO_ROOT / ".cache"
DOCS_DATA = REPO_ROOT / "docs" / "data"


def _ensure(directory):
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def figure(name):
    """Absolute path of a figure in ``figures/``."""
    return _ensure(FIGURES) / name


def result(name):
    """Absolute path of a table in ``results/``."""
    return _ensure(RESULTS) / name


def cache(name):
    """Absolute path of a regenerable artefact in ``.cache/``."""
    return _ensure(CACHE) / name
