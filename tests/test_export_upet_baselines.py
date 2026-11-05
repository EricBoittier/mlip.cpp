#!/usr/bin/env python3
"""Exporting a upet checkpoint by name must keep its fitted baselines (energy scale, composition energies).
LLPR-wrapped upet checkpoints (e.g. pet-mols-s) used to lose them for `--model <name>`: energy scale 1.0 and no
composition, i.e. forces 1/scale too large. The name and the local-checkpoint export must write the same baselines.

    uv run pytest tests/test_export_upet_baselines.py -v   (needs metatrain, upet checkpoints via HF or the HF cache)
"""
import os, subprocess, sys
import pytest

gguf = pytest.importorskip("gguf")
pytest.importorskip("metatrain")
HERE = os.path.dirname(os.path.abspath(__file__))
EXPORT = os.path.join(HERE, "..", "scripts", "export_pytorch", "export_pet_gguf.py")
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
MODEL = os.environ.get("MLIPCPP_TEST_UPET", "pet-mols-s")


def baselines(path):
    r = gguf.GGUFReader(path)
    val = lambda k: r.fields[k].parts[r.fields[k].data[0]] if k in r.fields else None
    keys = r.fields.get("pet.composition_keys")
    comp = list(keys.data) if keys is not None else []
    return float(val("pet.energy_scale")[0]), len(comp)


def test_name_and_checkpoint_exports_match(tmp_path):
    from export_pytorch.export_pet_full import upet_checkpoint_path
    ckpt = upet_checkpoint_path(MODEL)
    out = {}
    for tag, src in (("name", MODEL), ("ckpt", ckpt)):
        f = tmp_path / f"{tag}.gguf"
        subprocess.run([sys.executable, EXPORT, "--model", src, "-o", str(f)], check=True, capture_output=True)
        out[tag] = baselines(str(f))
    assert out["name"] == out["ckpt"]
    scale, ncomp = out["name"]
    assert ncomp > 0 and scale != 1.0, out
