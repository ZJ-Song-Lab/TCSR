"""Regenerate the paper's data figures from the archived result tables.

Usage (repository root)::

    # the two figures whose inputs are fully archived under results/
    python scripts/plot_figures.py

    # also redraw the gap-stratified mechanism figure from a diagnostics run
    python scripts/plot_figures.py --stratified-json \
        outputs/hrsid_rtdetr/HR-02/seed512101/variant_G/diagnostics.json

    # redraw the paired-contrasts figure with the exact bootstrap CIs
    python scripts/plot_figures.py --contrast-csv outputs/hrsid_rtdetr/comparison.csv

Figures (written to ``--output-dir``, default ``outputs/figures``)::

    tcsr_paired_ap_gains.png          paper Fig. "paired" (tab:pairedmeans)
    stratified_mechanism_evidence.png paper Fig. "stratified" (diagnostics)
    tcsr_deployed_efficiency.png      paper Fig. "deployed" (deployment tables)

The paired figure plots the 16 prespecified contrasts (G - B, G - C_Q,
G - H, G - I over the four settings).  Point estimates come from
``results/tab_pairedmeans.csv``; the 95% confidence intervals require the
paired-comparison CSV produced by ``tools/method_comparison.py --csv``
(one file per setting, passed via ``--contrast-csv``).  The stratified
figure reads the ``stratified`` block of a ``tools/diagnostics.py`` JSON
report.  The deployed figure reads ``results/tab_deployment.csv`` and
``results/tab_engineclosure.csv``.

The published figure files are archived under ``figures/``; this script
regenerates them from the archived inputs (matplotlib only).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

#: settings in the paper's display order
SETTINGS: Tuple[str, ...] = ("HRSID/RT-DETR", "SSDD/RT-DETR",
                             "COCO/RT-DETR", "COCO/DefDETR")

#: the four prespecified contrasts (method - baseline)
CONTRASTS: Tuple[Tuple[str, str], ...] = (("G", "B"), ("G", "C_Q"),
                                          ("G", "H"), ("G", "I"))

CONTRAST_COLORS = {"G-B": "#1f77b4", "G-C_Q": "#ff7f0e",
                   "G-H": "#7f7f7f", "G-I": "#d62728"}

_VARIANT_ID_RE = re.compile(r"^(C_Q|[A-I])_")


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def variant_id(variant_column: str) -> str:
    """'G_tcsr' -> 'G', 'C_Q_gQ_ranked' -> 'C_Q'."""
    m = _VARIANT_ID_RE.match(variant_column.strip())
    if not m:
        raise ValueError(f"cannot parse variant id from {variant_column!r}")
    return m.group(1)


def load_paired_means(results_dir: str) -> Dict[Tuple[str, str], float]:
    """{(setting, variant_id): ap_mean} from tab_pairedmeans.csv."""
    rows = read_csv_rows(os.path.join(results_dir, "tab_pairedmeans.csv"))
    out: Dict[Tuple[str, str], float] = {}
    for r in rows:
        out[(r["setting"], variant_id(r["variant"]))] = float(r["ap_mean"])
    return out


def load_contrast_cis(paths: Sequence[str]) -> Dict[Tuple[str, str],
                                                     Tuple[float, float]]:
    """{(setting, 'G-B'): (ci_low, ci_high)} from method_comparison CSVs."""
    out: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for path in paths:
        for r in read_csv_rows(path):
            setting = r.get("group") or r.get("setting") or ""
            contrast = r.get("contrast", "")
            try:
                out[(setting, contrast)] = (float(r["ci_low"]),
                                            float(r["ci_high"]))
            except (KeyError, ValueError):
                continue
    return out


def _match_setting(cis, setting: str, contrast: str
                   ) -> Optional[Tuple[float, float]]:
    """CI lookup tolerant of group labels that abbreviate the setting."""
    for key, val in cis.items():
        key_setting, key_contrast = key
        if key_contrast == contrast and (key_setting == setting
                                         or key_setting.startswith(setting)
                                         or setting.startswith(key_setting)):
            return val
    return None


# ----------------------------------------------------------------------
# Fig. "paired": primary paired contrasts of G against B / C_Q / H / I.
# ----------------------------------------------------------------------

def plot_paired(results_dir: str, out_path: str, dpi: int = 200,
                contrast_csvs: Optional[Sequence[str]] = None) -> str:
    means = load_paired_means(results_dir)
    cis = load_contrast_cis(contrast_csvs or [])
    bar_h = 0.19
    fig, ax = plt.subplots(figsize=(9.0, 6.2))
    for gi, setting in enumerate(SETTINGS):
        y0 = -(gi * 1.0)
        for ci, (a, b) in enumerate(CONTRASTS):
            key = (setting, a)
            if key not in means or (setting, b) not in means:
                raise KeyError(f"tab_pairedmeans.csv: missing {a} or {b} "
                               f"for setting {setting!r}")
            diff = means[key] - means[(setting, b)]
            y = y0 + (1.5 - ci) * bar_h
            name = f"{a}-{b}"
            ci95 = _match_setting(cis, setting, name)
            ax.barh(y, diff, height=bar_h * 0.86,
                    color=CONTRAST_COLORS[name], edgecolor="black",
                    linewidth=0.4,
                    xerr=None if ci95 is None else
                    [[diff - ci95[0]], [ci95[1] - diff]],
                    error_kw=dict(ecolor="black", capsize=2.5, lw=1.0))
            ax.text(diff, y, f" {diff:+.2f}", va="center",
                    ha="left" if diff >= 0 else "right", fontsize=8.5)
        ax.text(0.005, y0 + 2.4 * bar_h + 0.06, setting, fontsize=9.5,
                fontweight="bold", transform=ax.get_yaxis_transform(),
                va="bottom")
    ax.set_yticks([])
    ax.axvline(0.0, color="black", lw=0.8)
    ax.set_xlabel("AP difference vs. baseline (points)")
    ax.set_title("Paired AP contrasts of TCSR (G) against the controls "
                 "B, C$_Q$, H, I", fontsize=11)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CONTRAST_COLORS[f"{a}-{b}"])
               for a, b in CONTRASTS]
    ax.legend(handles, [f"{a} $-$ {b}" for a, b in CONTRASTS],
              loc="lower right", fontsize=9, frameon=False)
    if not cis:
        ax.text(0.99, 0.02,
                "mean differences from tab_pairedmeans.csv "
                "(95% CIs need --contrast-csv)",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7.5, color="0.35")
    ax.set_ylim(-len(SETTINGS) + 0.4, 1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------
# Fig. "stratified": gap-stratified mechanism evidence (3 x 3 cells).
# ----------------------------------------------------------------------

def _extract_stratified(payload: Dict) -> Dict:
    """Accept either a diagnostics report or a raw gap_stratified dict."""
    if isinstance(payload.get("stratified"), dict):
        payload = payload["stratified"]
    if "cells" not in payload:
        raise KeyError("no 'stratified' block with 'cells' in the JSON "
                       "(run tools/diagnostics.py first)")
    return payload


def _grid(cells: Sequence[Sequence[Dict]], key: str) -> np.ndarray:
    return np.array([[float(cell.get(key, float("nan")))
                      for cell in row] for row in cells], dtype=float)


def plot_stratified(stratified_json: str, out_path: str, dpi: int = 200
                    ) -> str:
    with open(stratified_json, "r", encoding="utf-8-sig") as f:
        payload = _extract_stratified(json.load(f))
    cells = payload["cells"]
    c = _grid(cells, "mean_c")
    u0 = _grid(cells, "mean_u0")
    ndr = _grid(cells, "mean_neg_delta_r")
    n = _grid(cells, "n")
    labels = ("low", "medium", "high")
    panels = (("mean corroboration $c$", c, False),
              ("mean priority $u^0$", u0, False),
              ("residual reduction $-\\Delta r$", ndr, True))
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.9))
    for ax, (title, grid, show_n) in zip(axes, panels):
        im = ax.imshow(grid, cmap="viridis", aspect="auto")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if show_n:
                    txt = f"{grid[i, j]:.3f}\n(n={int(n[i, j]):,})"
                else:
                    txt = f"{grid[i, j]:.3f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="white" if grid[i, j] < np.nanmax(grid) * 0.7
                        else "black")
        if show_n and np.isfinite(ndr).any():
            ij = np.unravel_index(np.nanargmax(ndr), ndr.shape)
            ax.add_patch(plt.Rectangle((ij[1] - 0.5, ij[0] - 0.5), 1, 1,
                                       fill=False, edgecolor="red", lw=2.0))
        ax.set_xticks(range(3), labels)
        ax.set_yticks(range(3), labels)
        ax.set_xlabel("$g_Q$ tertile (quantization damage)")
        if ax is axes[0]:
            ax.set_ylabel("$g_D$ tertile (inherited discrepancy)")
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.suptitle("Gap-stratified mechanism analysis "
                 "(tertile edges frozen on the dev split)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------
# Fig. "deployed": accuracy-efficiency trade-off of the TRT engines.
# ----------------------------------------------------------------------

_ENGINE_LABELS = {"tensorrt_fp16_P": "FP16 P", "tensorrt_B": "B",
                  "tensorrt_G": "G"}
_ENGINE_COLORS = {"tensorrt_fp16_P": "#9467bd", "tensorrt_B": "#7f7f7f",
                  "tensorrt_G": "#1f77b4"}


def _deployment_rows(results_dir: str) -> Dict[str, Dict[str, str]]:
    rows = read_csv_rows(os.path.join(results_dir, "tab_deployment.csv"))
    out: Dict[str, Dict[str, str]] = {}
    for r in rows:
        if r["measurement_boundary"] == "model_only":
            out[r["model"]] = r
    return out


def _engine_aps(results_dir: str) -> Dict[str, float]:
    rows = read_csv_rows(os.path.join(results_dir, "tab_engineclosure.csv"))
    role_to_key = {"fp_engine": "tensorrt_fp16_P", "b_engine": "tensorrt_B",
                   "g_engine": "tensorrt_G"}
    out: Dict[str, float] = {}
    for r in rows:
        if r["role"] in role_to_key:
            out[role_to_key[r["role"]]] = float(r["ap"])
    return out


def plot_deployed(results_dir: str, out_path: str, dpi: int = 200) -> str:
    rows = _deployment_rows(results_dir)
    aps = _engine_aps(results_dir)
    keys = [k for k in ("tensorrt_fp16_P", "tensorrt_B", "tensorrt_G")
            if k in rows]
    if not keys:
        raise FileNotFoundError("tab_deployment.csv has no model_only rows")
    lat = np.array([float(rows[k]["latency_p50_ms"]) for k in keys])
    energy = np.array([float(rows[k]["energy_j_per_image"]) for k in keys])
    mem = np.array([float(rows[k]["peak_memory_gib"]) for k in keys])
    labels = [_ENGINE_LABELS[k] for k in keys]
    colors = [_ENGINE_COLORS[k] for k in keys]

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    # (a) AP vs median latency
    ax = axes[0]
    ax.scatter(lat, [aps.get(k, float("nan")) for k in keys], s=90,
               c=colors, edgecolors="black", zorder=3)
    for k, x in zip(keys, lat):
        ax.annotate(_ENGINE_LABELS[k], (x, aps.get(k, float("nan"))),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    if "tensorrt_B" in aps and "tensorrt_G" in aps:
        ax.annotate(f"G $-$ B = {aps['tensorrt_G'] - aps['tensorrt_B']:+.2f} AP",
                    (0.03, 0.06), xycoords="axes fraction", fontsize=9)
    ax.set_xlabel("median latency p50 (ms, model-only)")
    ax.set_ylabel("engine AP")
    ax.set_title("(a) AP vs. latency", fontsize=10)
    ax.grid(alpha=0.3, zorder=0)
    # (b) energy per image
    ax = axes[1]
    ax.bar(labels, energy, color=colors, edgecolor="black", linewidth=0.5)
    if "tensorrt_G" in rows and "tensorrt_fp16_P" in rows:
        e_g = float(rows["tensorrt_G"]["energy_j_per_image"])
        e_p = float(rows["tensorrt_fp16_P"]["energy_j_per_image"])
        ax.annotate(f"G vs FP16: {(e_g / e_p - 1) * 100:+.1f}%",
                    (0.03, 0.9), xycoords="axes fraction", fontsize=9)
    for i, v in enumerate(energy):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylabel("energy per image (J)")
    ax.set_title("(b) energy per image", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    # (c) peak memory
    ax = axes[2]
    ax.bar(labels, mem, color=colors, edgecolor="black", linewidth=0.5)
    if "tensorrt_G" in rows and "tensorrt_fp16_P" in rows:
        m_g = float(rows["tensorrt_G"]["peak_memory_gib"])
        m_p = float(rows["tensorrt_fp16_P"]["peak_memory_gib"])
        ax.annotate(f"G vs FP16: {(m_g / m_p - 1) * 100:+.1f}%",
                    (0.03, 0.9), xycoords="axes fraction", fontsize=9)
    for i, v in enumerate(mem):
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_ylabel("peak memory (GiB)")
    ax.set_title("(c) peak memory", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Deployed accuracy-efficiency of the TensorRT engines "
                 "(HRSID/HR-02, NVIDIA L4, model-only)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ----------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results-dir", default="results",
                   help="directory holding the archived tab_*.csv tables")
    p.add_argument("--output-dir", default=os.path.join("outputs", "figures"),
                   help="directory for the regenerated PNGs")
    p.add_argument("--contrast-csv", action="append", default=None,
                   help="paired-comparison CSV of tools/method_comparison.py "
                        "(repeatable; adds 95% CIs to the paired figure)")
    p.add_argument("--stratified-json", default=None,
                   help="diagnostics.json of tools/diagnostics.py "
                        "(required for the stratified figure)")
    p.add_argument("--figures", default="paired,deployed",
                   help="comma list of: paired, stratified, deployed "
                        "(default: the figures with archived inputs)")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    requested = [f.strip() for f in args.figures.split(",") if f.strip()]
    known = ("paired", "stratified", "deployed")
    unknown = [f for f in requested if f not in known]
    if unknown:
        raise SystemExit(f"--figures: unknown {unknown}; expected {known}")
    os.makedirs(args.output_dir, exist_ok=True)
    written: List[str] = []
    for fig in requested:
        if fig == "paired":
            out = os.path.join(args.output_dir, "tcsr_paired_ap_gains.png")
            plot_paired(args.results_dir, out, dpi=args.dpi,
                        contrast_csvs=args.contrast_csv)
        elif fig == "stratified":
            if not args.stratified_json:
                raise SystemExit("the stratified figure needs "
                                 "--stratified-json (run tools/diagnostics.py)")
            out = os.path.join(args.output_dir,
                               "stratified_mechanism_evidence.png")
            plot_stratified(args.stratified_json, out, dpi=args.dpi)
        else:
            out = os.path.join(args.output_dir,
                               "tcsr_deployed_efficiency.png")
            plot_deployed(args.results_dir, out, dpi=args.dpi)
        written.append(out)
        print(f"[plot_figures] wrote {out}")
    if "stratified" not in requested:
        print("[plot_figures] note: 'stratified' needs a diagnostics JSON "
              "(--stratified-json); the published figure is archived under "
              "figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
