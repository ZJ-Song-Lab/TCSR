"""Published-results CSV integrity tests (paper-reported endpoints)."""

from pathlib import Path

import pandas as pd
import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results"

EXPECTED_FILES = {
    "tab_main.csv", "tab_ablation.csv", "tab_baselines.csv",
    "tab_bitsweep.csv", "tab_conflict.csv", "tab_cost.csv",
    "tab_deployment.csv", "tab_engineclosure.csv", "tab_errorchain.csv",
    "tab_gapstats.csv", "tab_models.csv", "tab_pairedmeans.csv",
    "tab_pairrobust.csv", "tab_provenanceaudit.csv", "tab_sensitivity.csv",
    "tab_transferrobust.csv",
}


def test_all_tables_present():
    files = {p.name for p in RESULTS.glob("*.csv")}
    assert EXPECTED_FILES <= files


def test_main_table_columns():
    df = pd.read_csv(RESULTS / "tab_main.csv")
    assert list(df.columns) == ["setting", "method", "bits", "ap", "ap50",
                                "ap75", "ap_s", "delta_ap_vs_b"]
    assert len(df) == 14


def test_main_table_headline_numbers():
    df = pd.read_csv(RESULTS / "tab_main.csv")
    idx = df.set_index(["setting", "method"])

    def ap(setting, method):
        return float(idx.loc[(setting, method), "ap"])

    # HRSID / RT-DETR (paper Table: headline setting)
    assert ap("HRSID/RT-DETR", "full_precision_P") == pytest.approx(63.30)
    assert ap("HRSID/RT-DETR", "common_block") == pytest.approx(59.82)
    assert ap("HRSID/RT-DETR", "development_selected_ptq") == pytest.approx(61.39)
    assert ap("HRSID/RT-DETR", "B_student_only") == pytest.approx(61.91)
    assert ap("HRSID/RT-DETR", "G_tcsr") == pytest.approx(63.15)
    # other settings
    assert ap("SSDD/RT-DETR", "G_tcsr") == pytest.approx(60.31)
    assert ap("COCO/RT-DETR", "G_tcsr") == pytest.approx(46.16)
    assert ap("COCO/DefDETR", "G_tcsr") == pytest.approx(45.02)

    # G - B deltas reported in the last column
    for setting in idx.index.get_level_values(0).unique():
        d_b = float(idx.loc[(setting, "B_student_only"), "delta_ap_vs_b"])
        assert d_b == pytest.approx(0.0)
        g = ap(setting, "G_tcsr")
        b = ap(setting, "B_student_only")
        d_g = float(idx.loc[(setting, "G_tcsr"), "delta_ap_vs_b"])
        assert d_g == pytest.approx(g - b, abs=0.02)


def test_main_table_fp_beats_g_beats_b():
    df = pd.read_csv(RESULTS / "tab_main.csv")
    idx = df.set_index(["setting", "method"])["ap"]
    for setting in df["setting"].unique():
        fp = float(idx.loc[(setting, "full_precision_P")])
        g = float(idx.loc[(setting, "G_tcsr")])
        b = float(idx.loc[(setting, "B_student_only")])
        assert fp > g > b


def test_ablation_table_shape():
    df = pd.read_csv(RESULTS / "tab_ablation.csv")
    assert len(df) == 10
    assert set(df["variant_id"]).issubset(
        {"A", "B", "C", "C_Q", "D", "E", "F", "G", "H", "I"})


def test_tables_are_readable():
    for name in sorted(EXPECTED_FILES):
        df = pd.read_csv(RESULTS / name)
        assert not df.empty, name
