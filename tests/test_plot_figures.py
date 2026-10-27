"""scripts/plot_figures.py: regeneration of the paper's data figures."""

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
_SPEC = importlib.util.spec_from_file_location(
    "plot_figures", REPO / "scripts" / "plot_figures.py")
plot_figures = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(plot_figures)


def _run_main(tmp_path, figures, extra=()):
    out_dir = tmp_path / "figs"
    argv = ["--output-dir", str(out_dir), "--figures", figures,
            "--dpi", "80"] + list(extra)
    rc = plot_figures.main(argv)
    assert rc == 0
    return out_dir


def test_variant_id_parsing():
    assert plot_figures.variant_id("G_tcsr") == "G"
    assert plot_figures.variant_id("C_Q_gQ_ranked") == "C_Q"
    assert plot_figures.variant_id("B_student_only") == "B"
    assert plot_figures.variant_id("I_teacher_valued") == "I"
    with pytest.raises(ValueError):
        plot_figures.variant_id("unknown_label")


def test_paired_contrasts_match_paper():
    means = plot_figures.load_paired_means(str(RESULTS))
    hr = {v: m for (s, v), m in means.items() if s == "HRSID/RT-DETR"}
    assert hr["G"] - hr["B"] == pytest.approx(1.24, abs=0.005)
    assert hr["G"] - hr["C_Q"] == pytest.approx(0.74, abs=0.005)
    assert hr["G"] - hr["H"] == pytest.approx(1.15, abs=0.005)
    assert hr["G"] - hr["I"] == pytest.approx(1.43, abs=0.005)


def test_paired_figure_from_archived_results(tmp_path):
    out_dir = _run_main(tmp_path, "paired")
    out = out_dir / "tcsr_paired_ap_gains.png"
    assert out.exists() and out.stat().st_size > 5_000


def test_deployed_figure_from_archived_results(tmp_path):
    out_dir = _run_main(tmp_path, "deployed")
    out = out_dir / "tcsr_deployed_efficiency.png"
    assert out.exists() and out.stat().st_size > 5_000


def test_stratified_figure_from_diagnostics_fixture(tmp_path):
    def cell(n, c, u0, ndr):
        return {"n": n, "mean_c": c, "mean_u0": u0, "n_delta": n,
                "mean_neg_delta_r": ndr, "mean_correct": 0.9,
                "mean_best_iou": 0.7}
    cells = [[cell(3000 + 100 * i + 10 * j, 0.8 - 0.1 * i - 0.05 * j,
                   0.05 + 0.1 * j, 0.01 + 0.03 * j) for j in range(3)]
             for i in range(3)]
    payload = {"stratified": {"edges_g_D": [0.08, 0.15],
                              "edges_g_Q": [0.12, 0.22],
                              "cells": cells, "overall": cells[1][1],
                              "n_no_gt": 0}}
    fixture = tmp_path / "diagnostics.json"
    fixture.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "stratified_mechanism_evidence.png"
    plot_figures.plot_stratified(str(fixture), str(out), dpi=80)
    assert out.exists() and out.stat().st_size > 5_000


def test_extract_stratified_accepts_raw_gap_stratified_dict():
    raw = {"edges_g_D": [0.0, 1.0], "edges_g_Q": [0.0, 1.0],
           "cells": [[{"n": 1, "mean_c": 0.5, "mean_u0": 0.2,
                       "n_delta": 1, "mean_neg_delta_r": 0.1}] * 3] * 3,
           "overall": {}, "n_no_gt": 0}
    assert plot_figures._extract_stratified(raw)["cells"][0][0]["n"] == 1
    with pytest.raises(KeyError):
        plot_figures._extract_stratified({"unrelated": {}})


def test_contrast_ci_matching(tmp_path):
    csv_path = tmp_path / "comparison.csv"
    csv_path.write_text(
        "group,contrast,n,mean_diff,sd_diff,cohens_dz,ci_low,ci_high,"
        "p_randomization,p_holm\n"
        "HRSID/RT-DETR,G-B,60,1.24,0.12,10.3,1.186,1.294,0.0,0.0\n"
        "HRSID,G-C_Q,60,0.74,0.10,7.4,0.690,0.789,0.0,0.0\n",
        encoding="utf-8")
    cis = plot_figures.load_contrast_cis([str(csv_path)])
    assert cis[("HRSID/RT-DETR", "G-B")] == (1.186, 1.294)
    # abbreviated group labels still resolve
    hit = plot_figures._match_setting(cis, "HRSID/RT-DETR", "G-C_Q")
    assert hit == (0.690, 0.789)
    assert plot_figures._match_setting(cis, "COCO/RT-DETR", "G-B") is None


def test_main_requires_stratified_json():
    with pytest.raises(SystemExit):
        plot_figures.main(["--figures", "stratified", "--output-dir",
                           "unused"])


def test_main_rejects_unknown_figure():
    with pytest.raises(SystemExit):
        plot_figures.main(["--figures", "paired,nope", "--output-dir",
                           "unused"])
