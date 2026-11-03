"""PhysNet / SpookyPhysNet parity: mlip.cpp (ggml, float32) vs mmml (JAX) on the same weights.

Needs the mmml environment (JAX, e3x, flax, ase) with mmml on PYTHONPATH and the
mlipcpp Python module (build/lib + python/ on sys.path, or installed). Skips
otherwise. As a script it prints the comparison and can write it as JSON:

    JAX_PLATFORMS=cpu PYTHONPATH=$MMML:build/lib:python python tests/test_physnet_parity.py \
        --json parity.json [--checkpoint student.json ...]

Structures: water / methanol / octanol monomers, random dimers of them (2.2-6 A,
some inside the ZBL range), clusters from the octanol student's validation set,
and OMol25 validation molecules with their charge and spin multiplicity (Spooky).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "mmml"))
sys.path.insert(0, str(ROOT / "scripts"))

def _dev_package():
    """build/pydev/mlipcpp: python/mlipcpp/*.py and the built _mlipcpp module (uninstalled builds)."""
    so = sorted((ROOT / "build" / "lib").glob("_mlipcpp*.so"))
    if not so:
        return
    pkg = ROOT / "build" / "pydev" / "mlipcpp"
    pkg.mkdir(parents=True, exist_ok=True)
    for f in list((ROOT / "python" / "mlipcpp").glob("*.py")) + [so[0]]:
        link = pkg / f.name
        if not link.exists():
            link.symlink_to(f)
    sys.path.insert(0, str(pkg.parent))


_dev_package()

HOME = Path("/mmhome/boittier/home/metatomic-runs/training")
CHECKPOINTS = {
    "wm_student": HOME / "watmeoh/prod/students/best/wm_omol_l_A_cap1_mdaug_ft_ep59.json",
    "wmo_student": HOME / "octanol/students/wmo_ft3_al5g3.json",
}
MONOMERS = HOME / "octanol/monomers"
OCTANOL_VALID = HOME / "octanol/data/wmo_ft/valid.npz"
OMOL_VAL = HOME / "omol_universal/npz/val"

# targets (task): |dE| per atom < 1e-5 eV, max |dF| < 1e-4 eV/A
TOL_E_PER_ATOM = 1e-5
TOL_F = 1e-4


def _read_xyz(path):
    lines = Path(path).read_text().splitlines()
    n = int(lines[0])
    sym = [l.split()[0] for l in lines[2:2 + n]]
    R = np.array([[float(x) for x in l.split()[1:4]] for l in lines[2:2 + n]])
    from ase.data import atomic_numbers

    return np.array([atomic_numbers[s] for s in sym], np.int32), R


def _rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    a, b, c, d = q
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a-b*b+c*c-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a-b*b-c*c+d*d]])


def wmo_structures(n_dimers=16, n_clusters=6, seed=0, with_octanol=True):
    rng = np.random.default_rng(seed)
    mons = {"water": _read_xyz(MONOMERS / "water.xyz"), "methanol": _read_xyz(MONOMERS / "methanol.xyz")}
    if with_octanol:
        mons["octanol"] = _read_xyz(MONOMERS / "octanol.xyz")
    out = []
    for name, (Z, R) in mons.items():
        out.append(dict(name=name, Z=Z, R=R))
        out.append(dict(name=name + "_rattled", Z=Z, R=R + rng.normal(scale=0.05, size=R.shape)))
    names = list(mons)
    for k in range(n_dimers):
        a, b = names[k % len(names)], names[(k // len(names)) % len(names)]
        (Za, Ra), (Zb, Rb) = mons[a], mons[b]
        Ra = (Ra - Ra.mean(0)) @ _rotation(rng).T
        Rb = (Rb - Rb.mean(0)) @ _rotation(rng).T
        # closest heavy-atom contact at d (some inside the ZBL window)
        d = [1.1, 2.2, 2.8, 3.5, 4.5, 6.0, 9.0][k % 7]
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        proj_a, proj_b = (Ra @ u).max(), (Rb @ u).min()
        Rb = Rb + u * (proj_a - proj_b + d)
        out.append(dict(name=f"{a}-{b}_d{d}", Z=np.concatenate([Za, Zb]), R=np.concatenate([Ra, Rb])))
    if OCTANOL_VALID.exists() and n_clusters:
        d = np.load(OCTANOL_VALID, allow_pickle=True)
        idx = rng.choice(len(d["N"]), n_clusters, replace=False)
        for i in idx:
            n = int(d["N"][i])
            if not with_octanol and (d["Z"][i, :n] == 6).sum() > 1:
                continue
            out.append(dict(name=f"wmo_valid_{i}", Z=d["Z"][i, :n].astype(np.int32), R=d["R"][i, :n]))
    return out


def omol_structures(n=20, max_atoms=60, seed=0, max_z=87):
    files = sorted(OMOL_VAL.glob("shard_*.npz"))
    if not files:
        return []
    d = np.load(files[0], allow_pickle=True)
    rng = np.random.default_rng(seed)
    N, off = d["N"][:, 0], d["mol_offsets"]
    Q, S = d["Q"][:, 0], d["S"][:, 0]
    out = []
    # a mix of charges and multiplicities
    order = rng.permutation(len(N))
    want = {(q, s): 0 for q in (-2, -1, 0, 1, 2) for s in (1, 2, 3)}
    for i in order:
        if N[i] > max_atoms:
            continue
        Z = d["Z"][off[i]:off[i + 1]]
        if Z.max() > max_z:
            continue
        key = (int(Q[i]), int(S[i]))
        if want.get(key, 99) >= max(2, n // 6):
            continue
        want[key] = want.get(key, 0) + 1
        out.append(dict(name=f"omol_{i}_q{key[0]}_s{key[1]}", Z=Z.astype(np.int32), R=d["R"][off[i]:off[i + 1]],
                        Q=key[0], S=key[1]))
        if len(out) >= n:
            break
    return out


def compare(ckpt, structures, gguf_dir):
    import mlipcpp
    from convert_mmml import to_gguf
    from mmml_ref import JaxRef

    gguf = Path(gguf_dir) / (Path(ckpt).stem + ".gguf")
    to_gguf(str(ckpt), str(gguf))
    model = mlipcpp.Predictor(str(gguf))
    rows = []
    refs = {32: JaxRef(ckpt, x64=False), 64: None}
    # float64 reference in a subprocess-free way: x64 must be set before any array exists,
    # so it is computed by run_x64() below and merged
    E32, F32 = refs[32](structures)
    pos = np.concatenate([np.asarray(s["R"], np.float32) for s in structures])
    Zs = np.concatenate([np.asarray(s["Z"], np.int32) for s in structures])
    nat = np.array([len(s["Z"]) for s in structures], np.int32)
    q = np.array([int(s.get("Q", 0)) for s in structures], np.int32)
    sp = np.array([int(s.get("S", 1)) for s in structures], np.int32)
    Eb, Fb = model.predict_batch(pos, Zs, nat, q, sp, True)
    a0 = 0
    for k, s in enumerate(structures):
        n = len(s["Z"])
        r = model.predict(np.ascontiguousarray(np.asarray(s["R"], np.float32)), np.asarray(s["Z"], np.int32),
                          charge=int(s.get("Q", 0)), spin_multiplicity=int(s.get("S", 1)))
        rows.append(dict(name=s["name"], n_atoms=n, Q=int(s.get("Q", 0)), S=int(s.get("S", 1)),
                         E_jax32=float(E32[k]), E_cpp=float(r.energy), E_cpp_batch=float(Eb[k]),
                         dF32=float(np.abs(np.asarray(r.forces) - F32[k]).max()),
                         dF_batch=float(np.abs(Fb[a0:a0 + n] - F32[k]).max()),
                         F_cpp=np.asarray(r.forces, np.float64).tolist(), F_jax32=F32[k].tolist()))
        a0 += n
    return rows


def run_x64(ckpt, structures):
    """float64 JAX reference in a fresh interpreter (x64 is process-global)."""
    import pickle
    import subprocess

    with tempfile.TemporaryDirectory() as td:
        inp, outp = Path(td) / "in.pkl", Path(td) / "out.pkl"
        inp.write_bytes(pickle.dumps((str(ckpt), structures)))
        code = ("import pickle,sys; sys.path.insert(0,%r); from mmml_ref import JaxRef; "
                "c,s=pickle.load(open(%r,'rb')); r=JaxRef(c,x64=True); E,F=r(s); "
                "pickle.dump((E,F),open(%r,'wb'))") % (str(ROOT / "scripts" / "mmml"), str(inp), str(outp))
        subprocess.run([sys.executable, "-c", code], check=True, env=dict(os.environ, JAX_ENABLE_X64="1"))
        return pickle.loads(outp.read_bytes())


def summarise(rows, E64=None, F64=None):
    s = {}
    n = np.array([r["n_atoms"] for r in rows])
    e32 = np.array([r["E_jax32"] for r in rows])
    ec = np.array([r["E_cpp"] for r in rows])
    eb = np.array([r["E_cpp_batch"] for r in rows])
    s["n_structures"] = len(rows)
    s["max_dE_per_atom_vs_jax32"] = float(np.max(np.abs(ec - e32) / n))
    s["max_dE_vs_jax32"] = float(np.max(np.abs(ec - e32)))
    s["max_dF_vs_jax32"] = float(max(r["dF32"] for r in rows))
    s["max_dE_batch_vs_single"] = float(np.max(np.abs(eb - ec)))
    s["max_dF_batch_vs_jax32"] = float(max(r["dF_batch"] for r in rows))
    if E64 is not None:
        s["max_dE_per_atom_vs_jax64"] = float(np.max(np.abs(ec - E64) / n))
        s["max_dE_vs_jax64"] = float(np.max(np.abs(ec - E64)))
        s["max_dE_per_atom_jax32_vs_jax64"] = float(np.max(np.abs(e32 - E64) / n))
        s["max_dF_vs_jax64"] = float(max(np.abs(np.asarray(r["F_cpp"]) - F).max() for r, F in zip(rows, F64)))
        s["max_dF_jax32_vs_jax64"] = float(max(np.abs(np.asarray(r["F_jax32"]) - F).max() for r, F in zip(rows, F64)))
    s["max_abs_E"] = float(np.abs(e32).max())
    s["max_abs_F"] = float(max(np.abs(np.asarray(r["F_jax32"])).max() for r in rows))
    return s


def run(checkpoints: dict, spooky_random: bool = True, x64: bool = True):
    report = {}
    with tempfile.TemporaryDirectory() as td:
        if spooky_random:
            import subprocess

            ck = Path(td) / "spooky_random.json"
            subprocess.run([sys.executable, str(ROOT / "scripts/mmml/make_random_checkpoint.py"), "-o", str(ck)],
                           check=True, capture_output=True)
            # feature_rmsnorm (mmml SpookyPhysNet option, per-atom RMS-normalised refinement output)
            ck_rms = Path(td) / "spooky_random_rms.json"
            subprocess.run([sys.executable, str(ROOT / "scripts/mmml/make_random_checkpoint.py"), "--feature-rmsnorm",
                            "-o", str(ck_rms)], check=True, capture_output=True)
            # + charge_spin_per_atom (charge / spin inputs Q / N and (M - 1) / N per atom)
            ck_cspa = Path(td) / "spooky_random_rms_cspa.json"
            subprocess.run([sys.executable, str(ROOT / "scripts/mmml/make_random_checkpoint.py"), "--feature-rmsnorm",
                            "--charge-spin-per-atom", "-o", str(ck_cspa)], check=True, capture_output=True)
            checkpoints = dict(checkpoints, spooky_random=ck, spooky_random_rms=ck_rms, spooky_random_rms_cspa=ck_cspa)
        for name, ck in checkpoints.items():
            if name.startswith("spooky"):
                structs = omol_structures(20) + wmo_structures(n_dimers=4, n_clusters=0)
            else:
                structs = wmo_structures(with_octanol=True)
            rows = compare(ck, structs, td)
            E64, F64 = run_x64(ck, structs) if x64 else (None, None)
            report[name] = dict(summary=summarise(rows, E64, F64),
                                rows=[{k: v for k, v in r.items() if not k.startswith("F_")} for r in rows])
    return report


# --------------------------------------------------------------------------- pytest
def _available():
    try:
        import e3x  # noqa: F401
        import mlipcpp  # noqa: F401
        import mmml  # noqa: F401
    except Exception:
        return False
    return all(p.exists() for p in CHECKPOINTS.values())


def test_physnet_parity():
    import pytest

    if not _available():
        pytest.skip("needs mmml (JAX, e3x), the mlipcpp module and the student checkpoints")
    rep = run(dict(CHECKPOINTS), spooky_random=True, x64=True)
    for name, r in rep.items():
        s = r["summary"]
        assert s["n_structures"] >= 20, name
        assert s["max_dE_batch_vs_single"] < 1e-4, (name, s)
        if name.startswith("spooky_random"):
            # an untrained model's forces are large (round-off ~ |F| 1e-6): no worse than
            # mmml's own float32 evaluation against float64
            assert s["max_dE_per_atom_vs_jax64"] < TOL_E_PER_ATOM, (name, s)
            assert s["max_dF_vs_jax64"] < 2 * s["max_dF_jax32_vs_jax64"] + TOL_F, (name, s)
        else:
            assert s["max_dE_per_atom_vs_jax32"] < TOL_E_PER_ATOM, (name, s)
            assert s["max_dF_vs_jax32"] < TOL_F, (name, s)
            assert s["max_dF_batch_vs_jax32"] < TOL_F, (name, s)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", action="append", default=None)
    ap.add_argument("--no-x64", action="store_true")
    ap.add_argument("--no-spooky-random", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    cks = {Path(c).stem: Path(c) for c in a.checkpoint} if a.checkpoint else dict(CHECKPOINTS)
    rep = run(cks, spooky_random=not a.no_spooky_random, x64=not a.no_x64)
    for name, r in rep.items():
        print(name, json.dumps(r["summary"], indent=1))
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1))
