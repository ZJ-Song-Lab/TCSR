"""tools/operator_inventory.py: per-operator bit-width inventory."""

import importlib.util
import json
from pathlib import Path

import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "operator_inventory", REPO / "tools" / "operator_inventory.py")
operator_inventory = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(operator_inventory)


class _Demo(nn.Module):
    """Module names chosen to exercise every policy branch."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential()
        self.stem.conv = nn.Conv2d(3, 8, 1)
        self.encoder = nn.Sequential()
        self.encoder.proj = nn.Linear(8, 8)
        self.decoder = nn.Sequential()
        self.decoder.ffn = nn.Linear(8, 16)
        self.head = nn.Sequential()
        self.head.cls = nn.Linear(16, 2)


def test_inventory_rows_policy_branches():
    rows = operator_inventory.inventory_rows(
        _Demo(), "HRSID / RT-DETR", "demo", "mixed")
    by_name = {r["op_name"]: r for r in rows}
    assert by_name["stem.conv"]["fake_quant_bits"] == "FP16 (excluded)"
    assert by_name["stem.conv"]["engine_format"] == "FP16"
    assert by_name["head.cls"]["fake_quant_bits"] == "FP16 (excluded)"
    assert by_name["encoder.proj"]["fake_quant_bits"] == "W8A8"
    assert by_name["encoder.proj"]["engine_format"] == "W8A8 (Q/DQ)"
    # the mixed profile's decoder: fake-quant W4A8, engine weight-only INT4
    # with FP16 compute (W4A16)
    assert by_name["decoder.ffn"]["fake_quant_bits"] == "W4A8"
    assert by_name["decoder.ffn"]["engine_format"] == \
        "W4A16 (weight-only INT4, FP16 compute)"
    assert by_name["stem.conv"]["shape"] == "8x3x1"
    assert by_name["encoder.proj"]["shape"] == "8x8"


def test_inventory_rows_bit_sweep_profiles():
    model = _Demo()
    w8 = {r["op_name"]: r for r in operator_inventory.inventory_rows(
        model, "s", "demo", "w8a8")}
    assert w8["decoder.ffn"]["fake_quant_bits"] == "W8A8"
    w4 = {r["op_name"]: r for r in operator_inventory.inventory_rows(
        model, "s", "demo", "w4a4")}
    assert w4["encoder.proj"]["fake_quant_bits"] == "W4A4"


def test_cli_toy_model_writes_csv(tmp_path, capsys):
    out = tmp_path / "inventory.csv"
    argv = ["--config", str(REPO / "configs" / "hrsid_rtdetr.json"),
            "--model", "toy", "--seed", "0",
            "--output", str(out)]
    assert operator_inventory.main(argv) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == ("setting,model,profile,op_name,op_type,shape,"
                        "fake_quant_bits,engine_format")
    assert len(lines) > 3
    joined = "\n".join(lines)
    assert "FP16 (excluded)" in joined  # head ops stay full precision
    assert "W8A8 (Q/DQ)" in joined       # non-decoder operators
    # provenance sidecar next to the CSV
    sidecar = tmp_path / "inventory.csv.inventory.json"
    assert json.loads(sidecar.read_text(encoding="utf-8"))["mode"] == \
        "operator_inventory"
    assert "MACs shares" in capsys.readouterr().out


def test_cli_prints_table_without_output(capsys):
    argv = ["--config", str(REPO / "configs" / "hrsid_rtdetr.json"),
            "--model", "toy", "--seed", "1"]
    assert operator_inventory.main(argv) == 0
    out = capsys.readouterr().out
    assert "op_name" in out and "fake_quant_bits" in out


def test_cli_rejects_bad_profile():
    argv = ["--config", str(REPO / "configs" / "hrsid_rtdetr.json"),
            "--model", "toy", "--seed", "0", "--profile", "w3a9"]
    import pytest
    with pytest.raises(SystemExit):
        operator_inventory.main(argv)


def test_print_rows_empty(capsys):
    operator_inventory.print_rows([])
    assert "no Linear/Conv operators" in capsys.readouterr().out
