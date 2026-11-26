"""Shared utilities for the experiment scripts: paths, plotting style and table output."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # experiments parallelise across processes instead
os.environ.setdefault("PYTHONWARNINGS", "ignore::RuntimeWarning")  # inherited by joblib workers

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

# Categorical palette, fixed order (never cycled). Sequential palette: single blue ramp.
PALETTE = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f9ff", "#cce3fd", "#9dc8f7", "#6aabf1", "#338ced", "#2370cd", "#1757ac", "#0e3d7e"])
TEXT, TEXT2, GRID = "#191919", "#666666", "#e6e6e6"

QUICK = os.environ.get("RCEXPL_QUICK", "0") == "1"  # smoke-test mode: smaller runs


def style():
    if "seq_blue" not in matplotlib.colormaps:
        matplotlib.colormaps.register(SEQ)
        matplotlib.colormaps.register(SEQ.reversed(), name="seq_blue_r")
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.edgecolor": "#bdbdbd", "axes.labelcolor": TEXT, "xtick.color": TEXT2, "ytick.color": TEXT2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.6,
        "axes.prop_cycle": matplotlib.cycler(color=PALETTE), "legend.frameon": False,
        "image.cmap": "seq_blue",
    })


style()


def outdir(track: str, name: str) -> Path:
    p = RESULTS / track / name
    p.mkdir(parents=True, exist_ok=True)
    return p


def savefig(fig, path: Path):
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


def write_table(rows: list[dict], path: Path, floatfmt="{:.3f}"):
    """Write ``rows`` as both Markdown (``.md``) and CSV (``.csv``)."""
    if not rows:
        return
    cols = list(rows[0].keys())

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            if v != 0 and np.isfinite(v) and abs(v) < 1e-2:
                return f"{v:.2e}"
            return floatfmt.format(v)
        return str(v)

    md = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    md += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    path.with_suffix(".md").write_text("\n".join(md) + "\n")
    csv = [",".join(cols)] + [",".join(str(r[c]) for c in cols) for r in rows]
    path.with_suffix(".csv").write_text("\n".join(csv) + "\n")
    print(f"  saved {path.with_suffix('.md').relative_to(ROOT)}")


def write_json(obj, path: Path):
    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    path.write_text(json.dumps(obj, indent=2, default=default))


def standardize(data, n_train):
    mu, sd = data[:n_train].mean(0), data[:n_train].std(0)
    return (data - mu) / sd, mu, sd


class Timer:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        self.elapsed = time.time() - self.t


def mean_ci(x, z=1.96):
    x = np.asarray(x, float)
    return float(np.mean(x)), float(z * np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def csv_to_markdown(csv_path: Path, floatfmt="{:.3f}"):
    """Regenerate the Markdown table from a CSV written by :func:`write_table`."""
    lines = Path(csv_path).read_text().strip().splitlines()
    cols = lines[0].split(",")
    rows = []
    for line in lines[1:]:
        row = {}
        for c, v in zip(cols, line.split(",")):
            try:
                row[c] = float(v) if any(ch in v for ch in ".eE") or v.lstrip("-").isdigit() and "." in v else v
            except ValueError:
                row[c] = v
        rows.append(row)
    write_table(rows, Path(csv_path), floatfmt)
