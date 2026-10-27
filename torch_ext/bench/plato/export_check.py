"""Export a Plato checkpoint with attention_backend 'fused' via metatomic (the
mlipcpp library collected as an extension), then reload it in a fresh process
(the library loaded by metatomic only) and compare E, F with the eager manual model.

    python export_check.py save CKPT OUTDIR      # writes OUTDIR/model.pt, extensions/, ref.pt
    python export_check.py load OUTDIR
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import common  # noqa: E402

mode, dev = sys.argv[1], torch.device("cuda")
cc = common.cc()
ATOMS = {"dimer": [cc], "batch8": common.batch64()[:8], "cell160": [cc.repeat((2, 2, 2))]}

if mode == "save":
    import mlipcpp_torch

    ckpt, out = sys.argv[2], sys.argv[3]
    os.makedirs(out, exist_ok=True)
    mlipcpp_torch.load()
    ref = common.load(ckpt, torch.float32, dev)
    ref.set_attention_backend("manual")
    refs = {}
    for name, atoms in ATOMS.items():
        E, F = common.energy_forces(ref.eval(), common.make_systems(ref, atoms, torch.float32, dev))
        refs[name] = (E.cpu(), [f.cpu() for f in F])
    torch.save(refs, os.path.join(out, "ref.pt"))
    model = common.load(ckpt, torch.float32, "cpu")
    model.set_attention_backend("fused")
    assert model.use_fused
    exported = model.export()
    ext = os.path.join(out, "extensions")
    try:
        exported.save(os.path.join(out, "model.pt"), collect_extensions=ext)
        print("saved with collect_extensions", ext)
    except RuntimeError as e:
        print("collect_extensions failed:", e)
        exported.save(os.path.join(out, "model.pt"))
        print("saved without collecting (absolute extension path recorded)")
else:
    from metatomic.torch import load_atomistic_model

    out = sys.argv[2]
    assert "mlipcpp" not in str(torch.ops.loaded_libraries)
    ext = os.path.join(out, "extensions")
    model = load_atomistic_model(os.path.join(out, "model.pt"), extensions_directory=ext if os.path.isdir(ext) else None)
    print("loaded libraries:", torch.ops.loaded_libraries)
    model = model.to(dev)
    refs = torch.load(os.path.join(out, "ref.pt"))
    from metatomic.torch import ModelEvaluationOptions, ModelOutput

    opts = ModelEvaluationOptions(length_unit="angstrom", outputs={"energy": ModelOutput(sample_kind="system")})
    res = {}
    for name, atoms in ATOMS.items():
        systems = common.make_systems(model, atoms, torch.float32, dev)
        E = model(systems, opts, check_consistency=True)["energy"].block().values
        F = torch.autograd.grad(E.sum(), [s.positions for s in systems])
        E0, F0 = refs[name]
        F, F0 = -torch.cat(F).cpu(), torch.cat(F0)
        res[name] = {
            "E_rel": float((E.detach().cpu() - E0).abs().max() / E0.abs().max()),
            "F_rel": float((F - F0).abs().max() / F0.abs().max()),
        }
        print(name, res[name], flush=True)
    graph = str(getattr(model.module.layers, "0").fused_attention.graph)
    res["uses_mlipcpp_op"] = "mlipcpp::pet_attention" in graph
    res["op_registered_after_load"] = hasattr(torch.ops.mlipcpp, "pet_attention")
    print("exported graph calls mlipcpp::pet_attention:", res["uses_mlipcpp_op"])
    json.dump(res, open(os.path.join(out, "export_check.json"), "w"), indent=1)
