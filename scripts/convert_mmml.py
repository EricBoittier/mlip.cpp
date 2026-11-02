#!/usr/bin/env python3
"""Convert mmml PhysNet / SpookyPhysNet checkpoints to mlip.cpp GGUF and back.

    # mmml portable JSON ({"params", "config"}) or an Orbax run dir -> GGUF
    python scripts/convert_mmml.py to-gguf student.json -o student.gguf
    # GGUF (e.g. written by mlipcpp-train) -> mmml portable JSON
    python scripts/convert_mmml.py to-json student.gguf -o student_mlipcpp.json

The GGUF is a "pet-graph" model whose graph.json is one PHYSNET node
(src/models/physnet/physnet_graph.h), so GraphModel, mlipcpp-train and the
C/C++/Python APIs load it like any exported PET. All mmml hyperparameters are
stored as metadata (physnet.config: the full mmml config as JSON; the ones the
graph reads also as node params). JSON input needs only numpy; Orbax input
needs mmml (JAX, flax, orbax) on PYTHONPATH.

Unsupported checkpoint options (atomic charges / electrostatics, EFA,
PBC, trainable ZBL, max_degree > 1, n_res <= 0) are an error.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "mmml"))
from mmml_ref import dense_index, hypers, load_checkpoint, spooky_check_supported  # noqa: E402

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR = 4, 5, 6, 7, 8, 9
GGML_F32, GGML_I32 = 0, 4
ALIGN = 32

# tensors that are constants, not trained (excluded from train.param_names)
BUFFERS = ("zbl.z", "zbl.za", "cond.charge_values", "cond.spin_values")


# ---------------------------------------------------------------------------
# minimal GGUF v3 writer / reader (F32 / I32 tensors; string / i32 / f32 / array kv)
# ---------------------------------------------------------------------------
def _wstr(f, s: str):
    b = s.encode()
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def write_gguf(path: str | Path, kv: dict, tensors: dict[str, np.ndarray]):
    """kv values: str, int (i32), float (f32), bool, list[int] (i32), list[float] (f32)."""
    with open(path, "wb") as f:
        f.write(struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(tensors), len(kv)))
        for k, v in kv.items():
            _wstr(f, k)
            if isinstance(v, bool):
                f.write(struct.pack("<IB", T_BOOL, int(v)))
            elif isinstance(v, str):
                f.write(struct.pack("<I", T_STR))
                _wstr(f, v)
            elif isinstance(v, int):
                f.write(struct.pack("<Ii", T_I32, v))
            elif isinstance(v, float):
                f.write(struct.pack("<If", T_F32, v))
            elif isinstance(v, (list, tuple)):
                is_int = all(isinstance(x, (int, np.integer)) for x in v)
                f.write(struct.pack("<IIQ", T_ARR, T_I32 if is_int else T_F32, len(v)))
                f.write(np.asarray(v, np.int32 if is_int else np.float32).tobytes())
            else:
                raise TypeError(f"{k}: {type(v)}")
        offsets, off = [], 0
        blobs = []
        for name, a in tensors.items():
            a = np.ascontiguousarray(a)
            dt = GGML_I32 if a.dtype.kind in "iu" else GGML_F32
            a = a.astype(np.int32 if dt == GGML_I32 else np.float32)
            _wstr(f, name)
            ne = list(reversed(a.shape)) or [1]
            f.write(struct.pack("<I", len(ne)))
            for d in ne:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<IQ", dt, off))
            blobs.append(a.tobytes())
            off += (len(blobs[-1]) + ALIGN - 1) // ALIGN * ALIGN
        f.write(b"\0" * ((-f.tell()) % ALIGN))
        for b in blobs:
            f.write(b)
            f.write(b"\0" * ((-len(b)) % ALIGN))


def read_gguf(path: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    data = Path(path).read_bytes()
    pos = 0

    def rd(fmt):
        nonlocal pos
        v = struct.unpack_from("<" + fmt, data, pos)
        pos += struct.calcsize("<" + fmt)
        return v if len(v) > 1 else v[0]

    def rstr():
        nonlocal pos
        n = rd("Q")
        s = data[pos:pos + n].decode()
        pos += n
        return s

    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "B", 10: "Q", 11: "q", 12: "d"}

    def rval(t):
        if t == T_STR:
            return rstr()
        if t == T_ARR:
            et, n = rd("I"), rd("Q")
            return [rval(et) for _ in range(n)]
        v = rd(scalar[t])
        return bool(v) if t == T_BOOL else v

    magic, ver, nt, nkv = rd("IIQQ")
    if magic != GGUF_MAGIC:
        raise ValueError(f"{path}: not a GGUF file")
    kv = {}
    for _ in range(nkv):
        k = rstr()
        kv[k] = rval(rd("I"))
    info = []
    for _ in range(nt):
        name = rstr()
        nd = rd("I")
        ne = [rd("Q") for _ in range(nd)]
        dt, off = rd("IQ")
        info.append((name, ne, dt, off))
    align = int(kv.get("general.alignment", ALIGN))
    base = (pos + align - 1) // align * align
    tensors = {}
    for name, ne, dt, off in info:
        dtype = {GGML_F32: np.float32, GGML_I32: np.int32}[dt]
        count = int(np.prod(ne))
        a = np.frombuffer(data, dtype, count, base + off).reshape(list(reversed(ne)))
        tensors[name] = a.copy()
    return kv, tensors


# ---------------------------------------------------------------------------
# mmml params <-> GGUF tensors
# ---------------------------------------------------------------------------
def _f32(a):
    return np.asarray(a, np.float64).astype(np.float32)


def params_to_tensors(config: dict, params: dict, model_type: str, max_charge: int, max_spin: int) -> dict:
    h = hypers(config, model_type)
    F, T, R, Zmax = h["features"], h["num_iterations"], h["n_res"], h["max_atomic_number"]
    p = params
    t = {}
    emb = _f32(p["Embed_0"]["embedding"])  # (Z+1, 1, 1, F)
    t["embed"] = emb.reshape(emb.shape[0], F)
    if model_type == "spooky":
        for key, name in (("charge", "charge_feature_projection"), ("spin", "spin_feature_projection")):
            t[f"cond.{key}.w"] = _f32(p[name]["kernel"]).reshape(1, F).T  # numpy (F, 1) -> ne [1, F]
            t[f"cond.{key}.b"] = _f32(p[name]["bias"]).reshape(F)
        t["cond.charge_values"] = np.arange(-max_charge, max_charge + 1, dtype=np.float32).reshape(-1, 1)
        t["cond.spin_values"] = np.arange(1, max_spin + 1, dtype=np.float32).reshape(-1, 1)  # 2S+1
    for it in range(T):
        mp = p[f"MessagePass_{it}"]
        t[f"mp{it}.filter.0"] = _f32(mp["filter"]["0+"]["kernel"]).T  # (K, F) -> numpy (F, K)
        if "1-" in mp["filter"]:
            t[f"mp{it}.filter.1"] = _f32(mp["filter"]["1-"]["kernel"]).T
        k = _f32(mp["tensor"]["kernel"])  # (1, L1+1, 1, L2+1, 1, L3+1, F)
        if k.shape[0] != 1 or k.shape[2] != 1 or k.shape[4] != 1:
            raise ValueError("pseudotensor (parity 2) features are not supported")
        L1, L2, L3 = k.shape[1] - 1, k.shape[3] - 1, k.shape[5] - 1
        for l1 in range(L1 + 1):
            for l2 in range(L2 + 1):
                for l3 in range(L3 + 1):
                    if (l1, l2, l3) in ((0, 0, 0), (1, 1, 0), (0, 1, 1), (1, 0, 1)):
                        t[f"mp{it}.tp.{l1}{l2}{l3}"] = k[0, l1, 0, l2, 0, l3]
        for kk in range(R + 1):
            d = dense_index(h, it, kk)
            dn = p[f"Dense_{d}"]
            t[f"dense{d}.w0"] = _f32(dn["0+"]["kernel"]).T
            t[f"dense{d}.b0"] = _f32(dn["0+"]["bias"])
            if "1-" in dn:
                t[f"dense{d}.w1"] = _f32(dn["1-"]["kernel"]).T
    nd = T * (R + 1)
    t["energy.w"] = _f32(p[f"Dense_{nd}"]["0+"]["kernel"]).T  # (F, 1) -> (1, F)
    t["energy.scale"] = _f32(p[f"Dense_{nd + 1}"]["kernel"]).reshape(1)
    if h["use_energy_bias"]:
        t["energy_bias"] = _f32(p["energy_bias"]).reshape(-1, 1)
    z = np.arange(Zmax + 1, dtype=np.float64)
    t["zbl.z"] = z.astype(np.float32).reshape(-1, 1)
    t["zbl.za"] = (z ** 0.23).astype(np.float32).reshape(-1, 1)
    # sanity: every params leaf used
    used = {"Embed_0", "energy_bias", "charge_feature_projection", "spin_feature_projection"}
    used |= {f"MessagePass_{i}" for i in range(T)} | {f"Dense_{i}" for i in range(nd + 2)}
    extra = sorted(set(p) - used)
    if extra:
        raise ValueError(f"checkpoint has parameters mlip.cpp does not implement: {extra}")
    return t


def tensors_to_params(config: dict, model_type: str, t: dict) -> dict:
    h = hypers(config, model_type)
    F, T, R = h["features"], h["num_iterations"], h["n_res"]
    L = h["max_degree"]
    f64 = lambda a: np.asarray(a, np.float32)
    p = {"Embed_0": {"embedding": f64(t["embed"]).reshape(-1, 1, 1, F)}}
    if model_type == "spooky":
        for key, name in (("charge", "charge_feature_projection"), ("spin", "spin_feature_projection")):
            p[name] = {"kernel": f64(t[f"cond.{key}.w"]).T.reshape(1, F), "bias": f64(t[f"cond.{key}.b"])}
    for it in range(T):
        filt = {"0+": {"kernel": f64(t[f"mp{it}.filter.0"]).T}}
        if f"mp{it}.filter.1" in t:
            filt["1-"] = {"kernel": f64(t[f"mp{it}.filter.1"]).T}
        L1 = L
        L2 = 0 if it == 0 else L
        L3 = 0 if it == T - 1 else max(L1, L2)
        k = np.zeros((1, L1 + 1, 1, L2 + 1, 1, L3 + 1, F), np.float32)
        for key, a in t.items():
            pre = f"mp{it}.tp."
            if key.startswith(pre):
                l1, l2, l3 = (int(ch) for ch in key[len(pre):])
                k[0, l1, 0, l2, 0, l3] = a
        p[f"MessagePass_{it}"] = {"filter": filt, "tensor": {"kernel": k}}
        for kk in range(R + 1):
            d = dense_index(h, it, kk)
            dn = {"0+": {"kernel": f64(t[f"dense{d}.w0"]).T, "bias": f64(t[f"dense{d}.b0"])}}
            if f"dense{d}.w1" in t:
                dn["1-"] = {"kernel": f64(t[f"dense{d}.w1"]).T}
            p[f"Dense_{d}"] = dn
    nd = T * (R + 1)
    p[f"Dense_{nd}"] = {"0+": {"kernel": f64(t["energy.w"]).T}}
    p[f"Dense_{nd + 1}"] = {"kernel": f64(t["energy.scale"]).reshape(1, 1)}
    if "energy_bias" in t:
        p["energy_bias"] = f64(t["energy_bias"]).reshape(-1)
    return p


def to_gguf(src: str, out: str, model_type: str | None = None, max_charge: int = 10, max_spin: int = 10) -> dict:
    config, params, model_type = load_checkpoint(src, model_type)
    bad = spooky_check_supported(config, model_type)
    if bad:
        raise SystemExit(f"{src}: not supported by mlip.cpp PhysNet: " + "; ".join(bad))
    h = hypers(config, model_type)
    tensors = params_to_tensors(config, params, model_type, max_charge, max_spin)
    node_params = {k: h[k] for k in ("model_type", "features", "max_degree", "num_iterations", "num_basis_functions",
                                      "cutoff", "max_atomic_number", "zbl", "zbl_cutoff", "zbl_cuton")}
    node_params["n_res"] = h["n_res"]
    node_params["use_energy_bias"] = h["use_energy_bias"]
    inputs = ["input:species", "input:neighbor_species", "input:edge_vectors", "input:padding_mask",
              "input:reverse_neighbor_index"]
    gin = [
        {"name": "species", "dtype": "i32", "shape": ["n_atoms"]},
        {"name": "neighbor_species", "dtype": "i32", "shape": ["n_atoms", "max_neighbors"]},
        {"name": "edge_vectors", "dtype": "f32", "shape": ["n_atoms", "max_neighbors", 3]},
        {"name": "padding_mask", "dtype": "f32", "shape": ["n_atoms", "max_neighbors"]},
        {"name": "reverse_neighbor_index", "dtype": "i32", "shape": ["n_edges"]},
        {"name": "cutoff_values", "dtype": "f32", "shape": ["n_atoms", "max_neighbors"]},
    ]
    if model_type == "spooky":
        inputs += ["input:charge_index", "input:spin_index"]
        gin += [{"name": "charge_index", "dtype": "i32", "shape": ["n_atoms"]},
                {"name": "spin_index", "dtype": "i32", "shape": ["n_atoms"]}]
    graph = {
        "version": "1.0", "model_type": "physnet",
        "inputs": gin,
        "outputs": [{"name": "energy", "node_ref": "node:0", "dtype": "f32", "shape": ["n_atoms"]}],
        "nodes": [{"id": 0, "op": "PHYSNET", "name": "physnet", "inputs": inputs, "output_shape": ["n_atoms"],
                   "output_dtype": "f32", "params": node_params}],
    }
    Zmax = h["max_atomic_number"]
    species_map = []
    for z in range(1, Zmax + 1):
        species_map += [z, z]
    kv = {
        "general.architecture": "pet-graph",
        "general.name": Path(src).stem,
        "general.version": "1.0.2",
        "physnet.model_type": model_type,
        "physnet.config": json.dumps(config, default=str),
        "physnet.source": str(src),
        "pet.cutoff": float(h["cutoff"]),
        "pet.cutoff_width": 0.0,
        "pet.energy_scale": 1.0,
        "pet.forces_mode": 1,
        "pet.nc_forces": 0,
        "pet.num_neighbors_adaptive": 0.0,
        "pet.species_map": species_map,
        "pet.composition_keys": list(range(1, Zmax + 1)),
        "pet.composition_values": [0.0] * Zmax,
        "graph.json": json.dumps(graph),
        "train.param_names": json.dumps([k for k in tensors if k not in BUFFERS]),
        "graph.weight_shapes": json.dumps({k: list(v.shape) for k, v in tensors.items()}),
    }
    if model_type == "spooky":
        kv["pet.system_conditioning"] = 1
        kv["pet.max_charge"] = int(max_charge)
        kv["pet.max_spin_multiplicity"] = int(max_spin)
    write_gguf(out, kv, tensors)
    return {"model_type": model_type, "n_tensors": len(tensors),
            "n_params": int(sum(v.size for k, v in tensors.items() if k not in BUFFERS))}


def to_json(src: str, out: str, allow_baselines: bool = False) -> dict:
    kv, t = read_gguf(src)
    if "physnet.config" not in kv:
        raise SystemExit(f"{src}: not a PhysNet GGUF (no physnet.config)")
    config = json.loads(kv["physnet.config"])
    model_type = kv.get("physnet.model_type", "physnet")
    params = tensors_to_params(config, model_type, t)
    # baselines refitted by mlipcpp-train (without --model-baselines): fold into mmml's
    # parameters where they map exactly (composition -> energy_bias), else refuse
    scale = float(kv.get("pet.energy_scale", 1.0))
    comp = dict(zip(kv.get("pet.composition_keys", []), kv.get("pet.composition_values", [])))
    if abs(scale - 1.0) > 0 and not allow_baselines:
        raise SystemExit(f"{src}: pet.energy_scale = {scale} != 1 (scales ZBL too, no mmml equivalent); "
                         "train with --model-baselines or pass --allow-baselines to fold it approximately")
    if any(v != 0.0 for v in comp.values()):
        if "energy_bias" not in params:
            if not allow_baselines:
                raise SystemExit(f"{src}: composition energies but the model has no energy_bias "
                                 "(pass --allow-baselines to enable use_energy_bias in the exported config)")
            params["energy_bias"] = np.zeros(int(config.get("max_atomic_number", 118)) + 1, np.float32)
            config["use_energy_bias"] = True
        eb = np.asarray(params["energy_bias"], np.float64)
        for z, v in comp.items():
            eb[int(z)] += v / scale
        params["energy_bias"] = eb.astype(np.float32)
    if scale != 1.0:
        hh = hypers(config, model_type)
        params[f"Dense_{hh['num_iterations'] * (hh['n_res'] + 1) + 1}"]["kernel"] *= scale
        if "energy_bias" in params:
            params["energy_bias"] = params["energy_bias"] * scale
    if model_type == "spooky":
        config = dict(config, model_type="spooky")
    tolist = lambda x: {k: tolist(v) for k, v in x.items()} if isinstance(x, dict) else np.asarray(x).tolist()
    with open(out, "w") as f:
        json.dump({"params": tolist(params), "config": config}, f)
    return {"model_type": model_type, "n_tensors": len(t)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("to-gguf", help="mmml JSON / Orbax -> GGUF")
    a.add_argument("checkpoint")
    a.add_argument("-o", "--output", required=True)
    a.add_argument("--model-type", choices=["physnet", "spooky"], default=None)
    a.add_argument("--max-charge", type=int, default=10, help="Spooky: charge table range [-q, q]")
    a.add_argument("--max-spin-multiplicity", type=int, default=10, help="Spooky: 2S+1 table range [1, s]")
    b = sub.add_parser("to-json", help="GGUF -> mmml portable JSON")
    b.add_argument("gguf")
    b.add_argument("-o", "--output", required=True)
    b.add_argument("--allow-baselines", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "to-gguf":
        info = to_gguf(args.checkpoint, args.output, args.model_type, args.max_charge, args.max_spin_multiplicity)
    else:
        info = to_json(args.gguf, args.output, args.allow_baselines)
    print(json.dumps(info))


if __name__ == "__main__":
    main()
