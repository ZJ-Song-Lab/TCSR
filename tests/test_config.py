"""Config loading/inheritance and default-value tests."""

import json

import pytest

from tcsr.config import (FROZEN_MMD_BANDWIDTH, ExperimentConfig,
                         MethodConfig, default_configs, load_config)


class TestMethodConfigDefaults:
    def test_defaults(self):
        cfg = MethodConfig()
        assert (cfg.k_p, cfg.tau_p) == (30, 0.25)
        assert (cfg.k_t, cfg.tau_t) == (50, 0.30)
        assert cfg.tau_d == 0.35
        assert (cfg.eta_cls, cfg.eta_l1, cfg.eta_giou) == (0.50, 0.25, 0.25)
        assert cfg.beta == 0.75 and cfg.lambda_cor == 0.60
        assert (cfg.lambda_blk, cfg.lambda_fg, cfg.lambda_res,
                cfg.lambda_round) == (1.0, 1.0, 0.20, 0.01)
        assert cfg.stage1_lr == 2e-4 and cfg.stage2_lr == 5e-5
        assert cfg.stage1_updates_per_block == 6000
        assert cfg.stage2_updates == 8000
        assert cfg.sigma_min == 0.020

    def test_eta_sum_one(self):
        cfg = MethodConfig()
        assert cfg.eta_cls + cfg.eta_l1 + cfg.eta_giou == pytest.approx(1.0)


class TestExperimentConfig:
    def test_defaults(self):
        exp = ExperimentConfig()
        assert exp.name == "hrsid_rtdetr"
        assert exp.quant_profile == "mixed"
        assert exp.calib_images == 512
        assert exp.variants == ["A", "B", "C", "C_Q", "D", "E", "F", "G",
                                "H", "I"]

    def test_to_dict_and_save(self, tmp_path):
        exp = ExperimentConfig(name="unit_test")
        d = exp.to_dict()
        assert d["name"] == "unit_test"
        assert isinstance(d["method"], dict)
        p = tmp_path / "cfg.json"
        exp.save(str(p))
        loaded = json.loads(p.read_text(encoding="utf-8"))
        assert loaded["name"] == "unit_test"


class TestLoadConfig:
    def _write(self, tmp_path, name, payload):
        p = tmp_path / name
        p.write_text(json.dumps(payload), encoding="utf-8")
        return str(p)

    def test_load_simple(self, tmp_path):
        path = self._write(tmp_path, "a.json",
                           {"name": "x", "dataset": "ssdd"})
        cfg = load_config(path)
        assert cfg.name == "x" and cfg.dataset == "ssdd"
        # method defaults filled
        assert cfg.method.beta == 0.75

    def test_method_merge(self, tmp_path):
        path = self._write(tmp_path, "a.json",
                           {"method": {"beta": 0.5}})
        cfg = load_config(path)
        assert cfg.method.beta == 0.5
        assert cfg.method.lambda_cor == 0.60   # untouched default

    def test_base_inheritance_and_override(self, tmp_path):
        self._write(tmp_path, "base.json",
                    {"name": "base", "dataset": "hrsid",
                     "method": {"beta": 0.75, "lambda_cor": 0.6}})
        child = self._write(tmp_path, "child.json",
                            {"base": "base.json", "name": "child",
                             "method": {"beta": 0.9}})
        cfg = load_config(child)
        assert cfg.name == "child"        # overridden
        assert cfg.dataset == "hrsid"     # inherited
        assert cfg.method.beta == 0.9     # method merged per key
        assert cfg.method.lambda_cor == 0.6

    def test_unknown_keys_dropped(self, tmp_path):
        path = self._write(tmp_path, "a.json",
                           {"name": "x", "bogus_key": 123})
        cfg = load_config(path)
        assert not hasattr(cfg, "bogus_key")


class TestFrozenSettings:
    def test_frozen_bandwidths(self):
        assert FROZEN_MMD_BANDWIDTH == {
            "hrsid_rtdetr": 0.084, "ssdd_rtdetr": 0.079,
            "coco_rtdetr": 0.112, "coco_defdetr": 0.108}

    def test_default_configs(self):
        cfgs = default_configs()
        assert set(cfgs) == set(FROZEN_MMD_BANDWIDTH)
        for key, exp in cfgs.items():
            assert exp.mmd_bandwidth == FROZEN_MMD_BANDWIDTH[key]
        assert cfgs["hrsid_rtdetr"].head_type == "softmax"
        assert cfgs["coco_rtdetr"].head_type == "sigmoid"
        assert cfgs["coco_rtdetr"].num_classes == 80
        assert cfgs["coco_rtdetr"].calib_images == 1024
        assert cfgs["coco_defdetr"].model == "defdetr"
