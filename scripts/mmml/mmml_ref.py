"""mmml PhysNet / SpookyPhysNet helpers shared by the converter, tests and benchmarks.

- ``load_checkpoint(path)``: (config, flax params tree, model_type) from an mmml
  portable JSON (``{"params", "config"}``) or an Orbax run directory.
- ``JaxRef``: the mmml model itself (JAX, jit), evaluated with all intra-molecular
  pairs, as mmml's ML/MM does for monomers and dimers.
- ``numpy_energy``: an independent NumPy transcription of the math mlip.cpp
  implements (src/models/physnet/physnet_graph.cpp), for debugging parity.

Needs the mmml environment (JAX, e3x, flax) with mmml on PYTHONPATH.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

SUPPORTED_MODEL_TYPES = ("physnet", "spooky")


def _to_np(tree: Any) -> Any:
    if isinstance(tree, dict):
        return {k: _to_np(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)) and tree and isinstance(tree[0], (list, int, float)):
        return np.asarray(tree)
    if hasattr(tree, "shape"):
        return np.asarray(tree)
    return tree


def _peel(params: Any) -> dict:
    """The bare Flax variable tree (no 'params' wrappers)."""
    while isinstance(params, dict) and "params" in params and set(params) <= {"params", "config", "metadata", "model_attributes"}:
        params = params["params"]
    return params


def load_checkpoint(path: str | Path, model_type: str | None = None) -> tuple[dict, dict, str]:
    """Return (config, params, model_type) of an mmml PhysNet-family checkpoint.

    model_type: 'physnet' or 'spooky'; default: config['model_type'] if present,
    else 'spooky' when the params have charge/spin projections, else 'physnet'.
    """
    path = Path(path)
    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        config = data.get("config") or {}
        params = _peel(_to_np(data.get("params", data)))
        if not config and isinstance(data.get("params"), dict):
            config = data["params"].get("config") or data["params"].get("model_attributes") or {}
    else:  # Orbax run directory / epoch directory
        from mmml.models.physnetjax.physnetjax.restart.restart import get_last, get_params_model

        p = path
        if not (p / "_CHECKPOINT_METADATA").exists() and not p.name.startswith("epoch"):
            p = Path(get_last(str(p)))
        params, model = get_params_model(str(p), natoms=32, prefer_ema=True)
        params = _peel(_to_np(params))
        config = model.return_attributes() if model is not None else {}
        if model is not None and type(model).__name__ == "SpookyPhysNet":
            config = dict(config, model_type="spooky")
    if model_type is None:
        model_type = str(config.get("model_type", "")).lower()
        if model_type not in SUPPORTED_MODEL_TYPES:
            model_type = "spooky" if "charge_feature_projection" in params else "physnet"
    return dict(config), params, model_type


def build_model(config: dict, model_type: str, natoms: int = 64):
    from mmml.utils.model_checkpoint import build_physnet_from_config

    if model_type == "spooky":
        from mmml.models.physnetjax.physnetjax.models.spooky_model import SpookyPhysNet as cls
    else:
        from mmml.models.physnetjax.physnetjax.models.model import PhysNet as cls
    model = build_physnet_from_config(config, model_cls=cls, max_padded_atoms=natoms)
    if "zbl" in config:
        model.zbl = bool(config["zbl"])
    return model


def all_pairs(n: int, cutoff_positions: np.ndarray | None = None, cutoff: float | None = None):
    """dst, src of all ordered pairs i != j (optionally only those within cutoff)."""
    i, j = np.nonzero(~np.eye(n, dtype=bool))
    if cutoff is not None and cutoff_positions is not None:
        d = np.linalg.norm(cutoff_positions[j] - cutoff_positions[i], axis=1)
        keep = d < cutoff
        i, j = i[keep], j[keep]
    return i.astype(np.int32), j.astype(np.int32)


class JaxRef:
    """mmml's model evaluated on a batch of molecules (all pairs within each)."""

    def __init__(self, path: str | Path, *, x64: bool = False, model_type: str | None = None):
        import jax

        if x64:
            jax.config.update("jax_enable_x64", True)
        self.config, params, self.model_type = load_checkpoint(path, model_type)
        self.model = build_model(self.config, self.model_type)
        import jax.numpy as jnp

        dt = jnp.float64 if x64 else jnp.float32
        self.dtype = dt
        self.params = {"params": jax.tree_util.tree_map(lambda a: jnp.asarray(a, dt) if np.issubdtype(np.asarray(a).dtype, np.floating) else jnp.asarray(a), params)}
        self._fns = {}

    def _batch(self, mols):
        Z, R, seg, dst, src, Q, S = [], [], [], [], [], [], []
        off = 0
        for k, m in enumerate(mols):
            z = np.asarray(m["Z"], np.int32)
            n = len(z)
            i, j = all_pairs(n)
            Z.append(z), R.append(np.asarray(m["R"], np.float64)), seg.append(np.full(n, k, np.int32))
            dst.append(i + off), src.append(j + off)
            Q.append(np.full(n, float(m.get("Q", 0)))), S.append(np.full(n, float(m.get("S", 1))))
            off += n
        cat = np.concatenate
        return cat(Z), cat(R), cat(seg), cat(dst), cat(src), cat(Q), cat(S)

    def fn(self, n_atoms, n_pairs, n_mols, forces=True):
        import jax
        import jax.numpy as jnp

        key = (n_atoms, n_pairs, n_mols, forces)
        if key in self._fns:
            return self._fns[key]
        model = self.model

        def f(params, Z, R, seg, dst, src, Q, S):
            kw = dict(
                atomic_numbers=Z, positions=R, dst_idx=dst, src_idx=src, batch_segments=seg,
                batch_size=n_mols, batch_mask=jnp.ones(n_pairs, R.dtype), atom_mask=jnp.ones(n_atoms, R.dtype),
            )
            if self.model_type == "spooky":
                out = model.apply(params, Z, Q, S, R, dst, src, batch_segments=seg, batch_size=n_mols,
                                  batch_mask=kw["batch_mask"], atom_mask=kw["atom_mask"], compute_forces=forces)
            else:
                out = model.apply(params, compute_forces=forces, **kw)
            return out["energy"].reshape(-1), (out["forces"] if forces else None)

        self._fns[key] = jax.jit(f)
        return self._fns[key]

    def __call__(self, mols, forces=True):
        import jax.numpy as jnp

        Z, R, seg, dst, src, Q, S = self._batch(mols)
        f = self.fn(len(Z), len(dst), len(mols), forces)
        E, F = f(self.params, jnp.asarray(Z), jnp.asarray(R, self.dtype), jnp.asarray(seg), jnp.asarray(dst),
                 jnp.asarray(src), jnp.asarray(Q, self.dtype), jnp.asarray(S, self.dtype))
        E = np.asarray(E, np.float64)
        if F is None:
            return E, None
        F = np.asarray(F, np.float64)
        out, off = [], 0
        for m in mols:
            n = len(m["Z"])
            out.append(F[off:off + n])
            off += n
        return E, out


def spooky_check_supported(config: dict, model_type: str) -> list[str]:
    """Options of a checkpoint that mlip.cpp does not implement (empty: supported)."""
    bad = []
    if config.get("charges", False):
        bad.append("charges=True (atomic charges + electrostatics need all pairs to 10 A, beyond the cutoff)")
    if config.get("efa", False):
        bad.append("efa=True (Euclidean fast attention)")
    if config.get("use_pbc", False):
        bad.append("use_pbc=True")
    if int(config.get("n_refinement_blocks", config.get("n_res", 3))) <= 0:
        bad.append("n_refinement_blocks <= 0 (attention refinement / no residual block)")
    if int(config.get("max_degree", 1)) > 1:
        bad.append(f"max_degree={config.get('max_degree')} (only 0 and 1 are implemented)")
    if config.get("trainable_zbl", False):
        bad.append("trainable_zbl=True")
    if model_type == "spooky" and (config.get("interaction_trust_map", False)):
        bad.append("interaction_trust_map=True")
    if model_type != "spooky" and config.get("feature_rmsnorm", False):
        bad.append("feature_rmsnorm=True on PhysNet (a SpookyPhysNet option)")
    if model_type != "spooky" and config.get("charge_spin_per_atom", False):
        bad.append("charge_spin_per_atom=True on PhysNet (a SpookyPhysNet option)")
    return bad


# ---------------------------------------------------------------------------
# NumPy transcription of the mlip.cpp graph (physnet_graph.cpp), all pairs
# ---------------------------------------------------------------------------
BOHR_TO_ANGSTROM = 0.529177249
COULOMB_EV_ANGSTROM = 14.3996454784255
ZBL_PHI_C = np.array([0.18175, 0.50986, 0.28022, 0.02817])
ZBL_PHI_A = np.array([3.19980, 0.94229, 0.40290, 0.20162])


def hypers(config: dict, model_type: str) -> dict:
    T = int(config.get("num_iterations", 2))
    R = int(config.get("n_refinement_blocks", config.get("n_res", 3)))
    return dict(
        model_type=model_type, features=int(config.get("features", 32)), max_degree=int(config.get("max_degree", 1)),
        num_iterations=T, num_basis_functions=int(config.get("num_basis_functions", 16)), cutoff=float(config.get("cutoff", 6.0)),
        max_atomic_number=int(config.get("max_atomic_number", 118)), n_res=R, zbl=bool(config.get("zbl", True)),
        zbl_cutoff=float(config.get("zbl_cutoff", 0.6)),
        zbl_cuton=float(config["zbl_cuton"]) if config.get("zbl_cuton", 0.1) is not None else 0.0,
        use_energy_bias=bool(config.get("use_energy_bias", False)),
        feature_rmsnorm=bool(config.get("feature_rmsnorm", False)),
        charge_spin_per_atom=bool(config.get("charge_spin_per_atom", False)),
    )


def dense_index(h: dict, t: int, k: int) -> int:
    """Flax auto-name index of refinement Dense k (0..n_res) of iteration t."""
    return t * (h["n_res"] + 1) + k


def numpy_energy(config: dict, params: dict, model_type: str, Z, R, Q=0.0, S=1.0) -> float:
    h = hypers(config, model_type)
    Z = np.asarray(Z)
    R = np.asarray(R, np.float64)
    n = len(Z)
    i, j = all_pairs(n)
    v = R[j] - R[i]
    r = np.linalg.norm(v, axis=1)
    u = v / r[:, None]
    c = h["cutoff"]
    x2 = (r / c) ** 2
    cut = np.where(x2 < 1, np.exp(1 - 1 / (1 - np.minimum(x2, 1 - 1e-12))), 0.0)
    y = 2 * np.exp(-r) - 1
    K = h["num_basis_functions"]
    T = [np.ones_like(y), y]
    for _ in range(2, K):
        T.append(2 * y * T[-1] - T[-2])
    rbf = np.stack(T[:K], axis=1) * cut[:, None]  # [P, K]
    L = h["max_degree"]
    p = params
    s = p["Embed_0"]["embedding"][Z, 0, 0, :].astype(np.float64)  # [n, F]
    vec = None  # [n, 3, F]
    if model_type == "spooky":
        cq, cs = p["charge_feature_projection"], p["spin_feature_projection"]
        if h["charge_spin_per_atom"]:  # mmml: per-atom densities Q / N and (M - 1) / N, M = 2S+1
            Q, S = Q / n, (S - 1) / n
        s = s + Q * cq["kernel"][0] + cq["bias"] + S * cs["kernel"][0] + cs["bias"]
    sig = lambda a: 1 / (1 + np.exp(-a))

    def dense(name, s, vec, bias=True):
        d = p[name]
        s2 = s @ d["0+"]["kernel"] + (d["0+"]["bias"] if "bias" in d["0+"] else 0)
        v2 = None if vec is None else vec @ d["1-"]["kernel"]
        return s2, v2

    for t in range(h["num_iterations"]):
        mp = p[f"MessagePass_{t}"]
        f0 = rbf @ mp["filter"]["0+"]["kernel"]  # [P, F]
        a1 = None
        if L >= 1:
            a1 = u[:, :, None] * (rbf @ mp["filter"]["1-"]["kernel"])[:, None, :]  # [P,3,F]
        kern = mp["tensor"]["kernel"]  # (1, L1+1, 1, L2+1, 1, L3+1, F)
        L1, L2, L3 = kern.shape[1] - 1, kern.shape[3] - 1, kern.shape[5] - 1
        k = lambda l1, l2, l3: kern[0, l1, 0, l2, 0, l3]
        b0 = s[j]
        b1 = None if vec is None else vec[j]
        o0 = k(0, 0, 0) * f0 * b0
        if L1 >= 1 and L2 >= 1:
            o0 = o0 + k(1, 1, 0) * np.einsum("pcf,pcf->pf", a1, b1) / np.sqrt(3)
        o1 = None
        if L3 >= 1:
            o1 = np.zeros((len(i), 3, s.shape[1]))
            if L2 >= 1:
                o1 = o1 + k(0, 1, 1) * f0[:, None, :] * b1
            if L1 >= 1:
                o1 = o1 + k(1, 0, 1) * a1 * b0[:, None, :]
        s = np.zeros_like(s)
        np.add.at(s, i, o0)
        if o1 is not None:
            vec = np.zeros((n, 3, s.shape[1]))
            np.add.at(vec, i, o1)
        else:
            vec = None
        # refinement
        y_s = y_v = None
        for kk in range(h["n_res"]):
            g = sig(s)
            ys, yv = s + s * g, (None if vec is None else vec + vec * g[:, None, :])
            ys, yv = dense(f"Dense_{dense_index(h, t, kk)}", ys, yv)
            s, vec = s + ys, (None if vec is None else vec + yv)
            y_s, y_v = ys, yv
        y_s, y_v = dense(f"Dense_{dense_index(h, t, h['n_res'])}", y_s, y_v)
        g = sig(y_s)
        s, vec = y_s * g, (None if y_v is None else y_v * g[:, None, :])
        if h["feature_rmsnorm"]:  # one RMS per atom over every (degree, feature) component
            ms = (s * s).sum(1) + (0 if vec is None else (vec * vec).sum((1, 2)))
            inv = 1 / np.sqrt(ms / (s.shape[1] * (1 if vec is None else 4)) + 1e-6)
            s, vec = s * inv[:, None], (None if vec is None else vec * inv[:, None, None])
    nd = h["num_iterations"] * (h["n_res"] + 1)
    e = (s @ p[f"Dense_{nd}"]["0+"]["kernel"]) @ p[f"Dense_{nd + 1}"]["kernel"]
    e = e[:, 0]
    if h["use_energy_bias"]:
        e = e + p["energy_bias"][Z]
    E = e.sum()
    if h["zbl"]:
        zf = Z.astype(np.float64)
        za = zf ** 0.23
        a = 0.8854 * BOHR_TO_ANGSTROM / (za[i] + za[j])
        cc = ZBL_PHI_C / np.linalg.norm(ZBL_PHI_C)
        phi = (cc[None] * np.exp(-ZBL_PHI_A[None] * (r / a)[:, None])).sum(1)
        xs = np.clip((h["zbl_cutoff"] - r) / (h["zbl_cutoff"] - h["zbl_cuton"]), 0, 1)
        sw = ((6 * xs - 15) * xs + 10) * xs ** 3
        E += (0.5 * COULOMB_EV_ANGSTROM * zf[i] * zf[j] / r * phi * sw).sum()
    return float(E)
