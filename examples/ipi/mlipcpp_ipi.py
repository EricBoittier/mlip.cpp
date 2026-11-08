"""i-PI PES: mlip.cpp C API (libmlipcpp_c.so) forces, plus optional on-the-fly NMR shieldings (ShiftML4 through the
same C API), with online statistics, committee flags and a path-integral mass-scaling isotope estimator.

Forces only (as before):
  <ffdirect name="mlip"><pes>custom</pes><pes_path>mlipcpp_ipi.py</pes_path>
    <parameters>{template: start.xyz, model: pet-mols-s.gguf, lib: /path/libmlipcpp_c.so, backend: cuda}</parameters>
  </ffdirect>

With shieldings, give ffdirect batch_size = number of beads, so that one driver call receives the whole ring polymer:
  <ffdirect name="mlip" batch_size="4"><pes>custom</pes><pes_path>mlipcpp_ipi.py</pes_path>
    <parameters>{template: start.xyz, model: pet-mols-s.gguf, lib: ..., shield_models: "/m/shiftml4-*.gguf",
                 shield_stride: 80, shield_species: "1,6", temperature: 298, out: shield,
                 estimator: H2D, delta_model: pet-mols-s-delta.gguf, skip_steps: 8000, target_se: 0.03}</parameters>
  </ffdirect>

Parameters (all optional except template/model):
  shield_models   GGUFs of the committee (glob or comma list); one per member, per-atom output = sigma_iso in ppm
                  (scripts/export_pytorch/export_nanopet_gguf.py)
  shield_stride   evaluate shieldings every this many force calls (= MD steps) [1]
  shield_beads    all | centroid | k (every k-th bead, ring-polymer contraction of the observable) [all]
  shield_species  atomic numbers whose shieldings go to extras / statistics [6]
  extras_members  also return every member's sigma in extras (json key sigma_iso_members) [false]
  temperature     K, needed by the estimator [298]
  out             prefix of the plugin's own outputs: <out>_frames.npz (reduced per-frame data), <out>_online.json
                  (running estimates + errors), <out>_flagged.xyz (high-disagreement configurations); rewritten every
                  5 sampled frames, since i-PI may exit without running atexit handlers [shield]
  store_frames    keep the reduced per-frame arrays (tiny compared to trajectories) [true]
  skip_steps      discard the first steps from the online statistics [0]
  target_se       stop the run (write ./EXIT, i-PI soft exit) once the standard error of the stop quantity < this (ppm) [off]
  stop_on         fep | cumulant | scaling (estimator variants) | sigma (mean shielding of shield_species) [fep]
  min_steps       never stop before this many steps after skip_steps [0]
  uq_top          fraction of (frame, atom) committee spreads flagged as retraining candidates [0.01]
  uq_radius       radius (A) of the environment cluster written for a flagged atom [6.0]
  acf_lags        lags (in sampled frames) of the online autocorrelation of sigma; 0 = off [0]
  estimator       H2D: for every molecule containing H, the shieldings that molecule would have if its H were D,
                  by the scaled-coordinate free-energy perturbation (needs delta_model and batch mode) [off]
  delta_model     energy model for the estimator's potential differences: the force GGUF passed through
                  make_delta_gguf.py (zero composition, forward only), so per-atom energies are resolved to ~1e-7 eV
  est_dmin        minimum distance (A) between molecule centres scaled together in one evaluation [9.0]
  est_molecules   all | comma list of molecule indices to which the estimator applies (e.g. the protio molecules
                  of a mixed box) [all]
  m_from, m_to    masses (amu) of the substituted isotope in the simulation and in the target [1.00794, 2.01410] (i-PI's H and D)
  est_units       molecule: all H of a molecule substituted together (default) | each: every H atom on its own
                  (m_from -> m_to; e.g. CH2Cl2 -> CHDCl2) | "k:mfrom:mto; k:mfrom:mto ...": only the k-th hydrogen-type
                  atom (atom order) bonded to each heavy atom of the estimator molecules, with its own masses, one
                  class per entry ("/" may replace ":"; e.g. a CHDCl2 box: "1/1.00794/2.0141; 0/2.0141/1.00794"
                  contracts the H and expands the D)
                  [molecule]. In atom modes the stored sigma_t/lnw rows are units (est_sel, est_class, est_atom map them).
"""
import ctypes
import glob as _glob
import json
import os
import time

import numpy as np
from ase.data import chemical_symbols

try:
    from ipi.pes.dummy import Dummy_driver
    from ipi.utils.units import unit_to_internal, unit_to_user
except ImportError:  # offline use of ShieldingEngine without i-PI
    Dummy_driver = object
    unit_to_internal = unit_to_user = None

__DRIVER_NAME__ = "mlipcpp_ipi"
__DRIVER_CLASS__ = "MLIPCppDriver"

_BACKENDS = {"auto": 0, "cpu": 1, "cuda": 2, "hip": 3, "metal": 4, "vulkan": 5}
KB_EV = 8.617333262e-5
BOHR = 0.529177210903


class _ModelOptions(ctypes.Structure):
    _fields_ = [("backend", ctypes.c_int), ("precision", ctypes.c_int), ("cutoff_override", ctypes.c_float)]


class _PredictOptions(ctypes.Structure):
    _fields_ = [("compute_forces", ctypes.c_bool), ("compute_stress", ctypes.c_bool),
                ("use_nc_forces", ctypes.c_bool), ("charge", ctypes.c_int32),
                ("spin_multiplicity", ctypes.c_int32)]


def _flag(v):
    return str(v).lower() in ("1", "true", "yes", "on")


def _paths(spec):
    out = []
    for s in str(spec).replace(";", ",").split(","):
        s = s.strip()
        if s:
            out += sorted(_glob.glob(s)) or [s]
    return out


class CAPI:
    """Thin ctypes layer over libmlipcpp_c.so: load models, predict, read energy/forces/stress/per-atom values."""

    def __init__(self, lib_path, backend):
        lib = ctypes.CDLL(lib_path)
        lib.mlipcpp_set_backend.argtypes = [ctypes.c_int]
        lib.mlipcpp_model_options_default.argtypes = [ctypes.POINTER(_ModelOptions)]
        lib.mlipcpp_model_create.argtypes = [ctypes.POINTER(_ModelOptions)]
        lib.mlipcpp_model_create.restype = ctypes.c_void_p
        lib.mlipcpp_model_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.mlipcpp_predict_ptr_with_options.argtypes = [
            ctypes.c_void_p, ctypes.c_int32, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.POINTER(_PredictOptions), ctypes.POINTER(ctypes.c_void_p)]
        lib.mlipcpp_result_get_energy.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        lib.mlipcpp_result_get_forces.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int32]
        lib.mlipcpp_result_get_stress.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.mlipcpp_result_get_atomic_values.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.mlipcpp_result_free.argtypes = [ctypes.c_void_p]
        lib.mlipcpp_get_last_error.restype = ctypes.c_char_p
        self.backend = _BACKENDS[str(backend).lower()]
        lib.mlipcpp_set_backend(self.backend)
        self.lib = lib

    def load(self, path):
        opts = _ModelOptions()
        self.lib.mlipcpp_model_options_default(ctypes.byref(opts))
        opts.backend = self.backend
        h = self.lib.mlipcpp_model_create(ctypes.byref(opts))
        if not h or self.lib.mlipcpp_model_load(h, str(path).encode()) != 0:
            raise RuntimeError(f"mlip.cpp could not load {path}: {self.lib.mlipcpp_get_last_error()}")
        return h

    def predict(self, model, z, x, cell, pbc, popts):
        """x float32 (n,3) A, cell float32 (9,) rows = lattice vectors or None. Returns a result handle."""
        res = ctypes.c_void_p()
        st = self.lib.mlipcpp_predict_ptr_with_options(
            model, len(z), x.ctypes.data, z.ctypes.data, None if cell is None else cell.ctypes.data,
            None if cell is None else pbc.ctypes.data, ctypes.byref(popts), ctypes.byref(res))
        if st != 0:
            raise RuntimeError(self.lib.mlipcpp_get_last_error().decode())
        return res

    def atomic_values(self, model, z, x, cell, pbc, popts, out):
        res = self.predict(model, z, x, cell, pbc, popts)
        if self.lib.mlipcpp_result_get_atomic_values(res, out.ctypes.data) != 0:
            raise RuntimeError(self.lib.mlipcpp_get_last_error().decode())
        self.lib.mlipcpp_result_free(res)
        return out


# ------------------------------------------------------------------------------------------------- online statistics
class FPBlocking:
    """Flyvbjerg-Petersen blocking of a vector time series, online, O(log N) memory: level l holds blocks of 2^l
    samples. se(l) = sqrt(var_l / (n_l - 1)); the plateau over l is the standard error of the mean."""

    def __init__(self, dim, levels=24):
        self.dim, self.L = dim, levels
        self.n = np.zeros(levels, int)
        self.s = np.zeros((levels, dim))
        self.s2 = np.zeros((levels, dim))
        self.pend = [None] * levels

    def add(self, x):
        x = np.asarray(x, float).reshape(self.dim)
        for l in range(self.L):
            self.n[l] += 1
            self.s[l] += x
            self.s2[l] += x * x
            if self.pend[l] is None:
                self.pend[l] = x
                return
            x, self.pend[l] = 0.5 * (self.pend[l] + x), None

    @property
    def mean(self):
        return self.s[0] / max(self.n[0], 1)

    def se_levels(self):
        out = []
        for l in range(self.L):
            if self.n[l] < 8:
                break
            m = self.s[l] / self.n[l]
            v = self.s2[l] / self.n[l] - m * m
            out.append(np.sqrt(np.maximum(v, 0) / (self.n[l] - 1)))
        return np.array(out).reshape(-1, self.dim)

    def se(self):
        """Plateau estimate: the largest SE over levels with >= 16 blocks (conservative); NaN if too short."""
        lv = [l for l in range(self.L) if self.n[l] >= 16]
        if not lv:
            return np.full(self.dim, np.nan)
        return self.se_levels()[: lv[-1] + 1].max(0)


class BlockJackknife:
    """Online fixed-count blocking for nonlinear (ratio) estimators: at most kmax blocks of per-block sums; when full,
    neighbouring blocks merge and the block length doubles. Jackknife over blocks for any f(sums)."""

    def __init__(self, dim, kmax=64):
        self.dim, self.kmax, self.blen = dim, kmax, 1
        self.blocks, self.cur, self.ncur = [], np.zeros(dim), 0

    def add(self, x):
        self.cur += np.asarray(x, float).reshape(self.dim)
        self.ncur += 1
        if self.ncur == self.blen:
            self.blocks.append(self.cur)
            self.cur, self.ncur = np.zeros(self.dim), 0
            if len(self.blocks) == self.kmax:
                b = self.blocks
                self.blocks = [b[i] + b[i + 1] for i in range(0, len(b), 2)]
                self.blen *= 2

    def estimate(self, f):
        """f maps total sums (dim,) to an estimate (any shape). Returns (estimate, jackknife SE, n_blocks)."""
        if not self.blocks:
            return None, None, 0
        B = np.array(self.blocks)
        tot = B.sum(0)
        est = np.asarray(f(tot))
        k = len(B)
        if k < 4:
            return est, np.full(est.shape, np.nan), k
        jk = np.array([f(tot - b) for b in B])
        se = np.sqrt((k - 1) / k * ((jk - jk.mean(0)) ** 2).sum(0))
        return est, se, k


class OnlineACF:
    """<x(0) x(t)> for t = 0..lags-1 sampled frames, per component, averaged over components; ring buffer, O(lags*dim)."""

    def __init__(self, dim, lags):
        self.lags, self.buf, self.k = lags, np.zeros((lags, dim)), 0
        self.c, self.cn = np.zeros(lags), np.zeros(lags, int)
        self.s, self.n = np.zeros(dim), 0

    def add(self, x):
        x = np.asarray(x, float)
        self.buf[self.k % self.lags] = x
        m = min(self.k + 1, self.lags)
        for t in range(m):
            self.c[t] += float(np.dot(x, self.buf[(self.k - t) % self.lags])) / x.size
            self.cn[t] += 1
        self.s += x
        self.n += 1
        self.k += 1

    def result(self):
        mu = self.s / max(self.n, 1)
        ok = self.cn > 0
        return np.where(ok, self.c / np.maximum(self.cn, 1) - float(np.dot(mu, mu)) / mu.size, np.nan)


# ------------------------------------------------------------------------------------------------- shielding engine
class ShieldingEngine:
    """Committee shieldings on a ring polymer, the H->D scaled-coordinate estimator, online statistics, UQ flags."""

    def __init__(self, capi, numbers, template_positions, cell0, shield_models, species=(6,), beads_mode="all",
                 temperature=298.0, out="shield", store_frames=True, skip_steps=0, target_se=None, stop_on="dsigma",
                 min_steps=0, uq_top=0.01, uq_radius=6.0, acf_lags=0, estimator=None, delta_model=None,
                 est_dmin=9.0, est_molecules="all", m_from=1.00794, m_to=2.01410, extras_members=False, seed=0,
                 est_units="molecule"):
        self.capi = capi
        self.z = np.ascontiguousarray(numbers, dtype=np.int32)
        self.n = len(self.z)
        self.members = [capi.load(p) for p in shield_models]
        self.M = len(self.members)
        self.sel = np.where(np.isin(self.z, [int(s) for s in species]))[0]
        self.beads_mode, self.out = str(beads_mode), str(out)
        self.beta = 1.0 / (KB_EV * float(temperature))
        self.store, self.skip, self.min_steps = store_frames, int(skip_steps), int(min_steps)
        self.target = None if target_se in (None, "", "none", "off") else float(target_se)
        self.stop_on, self.uq_top, self.uq_radius = stop_on, float(uq_top), float(uq_radius)
        self.extras_members = extras_members
        self.popt = _PredictOptions(False, False, False, 0, 1)
        self.pbc = np.ones(3, dtype=np.bool_)
        self.buf = np.empty(self.n, dtype=np.float32)
        self.rng = np.random.default_rng(seed)
        self._molecules(np.asarray(template_positions, float), np.asarray(cell0, float))
        self.est = str(estimator).upper() in ("H2D", "TRUE", "1", "ON")
        if self.est:
            if not delta_model:
                raise ValueError("estimator needs delta_model (make_delta_gguf.py copy of the force model)")
            self.delta = capi.load(delta_model)
            self.alpha = float(np.sqrt(float(m_from) / float(m_to)))
            hm = [m for m in range(self.nmol) if (self.z[self.mol_atoms[m]] == 1).any()]
            if str(est_molecules) != "all":
                keep = set(int(v) for v in str(est_molecules).replace(";", ",").split(",") if v.strip())
                hm = [m for m in hm if m in keep]
            self.est_mols = np.array(hm, int)
            self.dmin = float(est_dmin)
            self.units = str(est_units).strip().lower()
            if self.units == "molecule":
                self.est_sel, self.est_class, self.nclass = np.arange(len(self.sel)), np.zeros(len(self.sel), int), 1
            else:  # atom units: (H atom, alpha, class)
                spec = ([(None, float(m_from), float(m_to))] if self.units == "each" else
                        [tuple(float(v) if i else int(v) for i, v in enumerate(e.split(":")))
                         for e in self.units.replace(",", ";").replace("/", ":").split(";") if e.strip()])
                # units per bonded heavy atom (robust to template molecules that the bond graph merges)
                ua, ual, ucl = [], [], []
                selpos = {int(a): j for j, a in enumerate(self.sel)}
                for c0 in sorted(set(self.hpar.tolist())):
                    if self.mol[c0] not in set(self.est_mols.tolist()):
                        continue
                    hs = sorted(int(h) for h, c in zip(self.hidx, self.hpar) if c == c0)
                    for c, (k, mf, mt) in enumerate(spec):
                        for a in (hs if k is None else [hs[k]]):
                            ua.append(a); ual.append(np.sqrt(mf / mt)); ucl.append(c)
                self.est_atom, self.est_alpha, self.est_class = np.array(ua), np.array(ual), np.array(ucl)
                self.est_sel = np.array([selpos.get(int(self.hpar[np.where(self.hidx == a)[0][0]]), -1) for a in ua])
                self.nclass = len(spec)
        # online statistics
        ns = len(self.sel)
        self.fp_sigma = FPBlocking(1)  # mean shielding of the selected atoms (committee mean)
        self.jk = [BlockJackknife(4 * self.M + 4) for _ in range(self.nclass)] if self.est else None
        self.acf = OnlineACF(ns, int(acf_lags)) if int(acf_lags) > 0 else None
        self.frames, self.spreads = [], []
        self.nflag, self.stopped, self.t0, self.nsampled = 0, False, time.time(), 0
        self.timing = {"shield_s": 0.0, "estimator_s": 0.0}
        self.extras = json.dumps({"shield_step": -1})
        import atexit

        atexit.register(self.flush)  # i-PI does not call the driver at exit

    # molecules from the template: bonded graph with covalent cutoffs (1.25x sum of radii), minimum image
    def _molecules(self, pos, cell):
        rad = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 15: 1.07, 16: 1.05, 17: 1.02, 35: 1.20}
        r = np.array([rad.get(int(a), 0.8) for a in self.z])
        inv = np.linalg.inv(cell)
        d = pos[:, None, :] - pos[None, :, :]
        d -= np.round(d @ inv) @ cell
        dist = np.linalg.norm(d, axis=-1)
        bond = (dist < 1.25 * (r[:, None] + r[None, :])) & (dist > 0.1)
        mol = -np.ones(self.n, int)
        k = 0
        for i in range(self.n):
            if mol[i] >= 0:
                continue
            stack = [i]
            mol[i] = k
            while stack:
                j = stack.pop()
                for q in np.where(bond[j] & (mol < 0))[0]:
                    mol[q] = k
                    stack.append(q)
            k += 1
        self.mol, self.nmol = mol, k
        self.mol_atoms = [np.where(mol == m)[0] for m in range(k)]
        # per H: bonded heavy atom; per heavy atom: the other heavy neighbours (bend axis)
        self.hidx = np.where(self.z == 1)[0]
        self.hpar = np.array([np.where(bond[h])[0][np.argmin(dist[h][bond[h]])] for h in self.hidx])
        self.heavy_nb = {int(c): np.where(bond[c] & (self.z != 1))[0] for c in set(self.hpar.tolist())}

    def _sigma(self, X, cell):
        """X (B, n, 3) A -> (B, M, n) shieldings."""
        out = np.empty((len(X), self.M, self.n), dtype=np.float32)
        for b, x in enumerate(X):
            xx = np.ascontiguousarray(x, dtype=np.float32)
            for k, h in enumerate(self.members):
                out[b, k] = self.capi.atomic_values(h, self.z, xx, cell, self.pbc, self.popt, self.buf)
        return out

    def _eatoms(self, X, cell):
        out = np.empty((len(X), self.n), dtype=np.float64)
        for b, x in enumerate(X):
            out[b] = self.capi.atomic_values(self.delta, self.z, np.ascontiguousarray(x, dtype=np.float32), cell,
                                             self.pbc, self.popt, self.buf)
        return out

    def _groups(self, centres, h):
        """Greedy colouring: molecules scaled together are >= dmin apart (minimum image)."""
        inv = np.linalg.inv(h)
        groups, cents = [], []
        for m in self.rng.permutation(self.est_mols):
            c = centres[m]
            for g, cg in zip(groups, cents):
                d = np.asarray(cg) - c
                d -= np.round(d @ inv) @ h
                if (np.linalg.norm(d, axis=1) >= self.dmin).all():
                    g.append(m)
                    cg.append(c)
                    break
            else:
                groups.append([m])
                cents.append([c])
        return groups

    def _geometry(self, X, h):
        """Per H: centroid C-H distance, bead-mean |r_CH|, H bead radius of gyration, bead-mean bend angle (deg)
        between C->H and the axis opposite the other heavy substituents, and the same angle of the centroids."""
        inv = np.linalg.inv(h)
        d = X[:, self.hidx] - X[:, self.hpar]  # (P, nH, 3); beads of one atom are not wrapped by i-PI
        d -= np.round(d @ inv) @ h
        dc = d.mean(0)
        hc = X[:, self.hidx].mean(0)
        rg = np.sqrt(((X[:, self.hidx] - hc) ** 2).sum(-1).mean(0))
        ang, angc = np.zeros(len(self.hidx)), np.zeros(len(self.hidx))
        for j, (hh, c) in enumerate(zip(self.hidx, self.hpar)):
            nb = self.heavy_nb[int(c)]
            if len(nb) == 0:
                continue
            v = X[:, nb] - X[:, [c]]
            v -= np.round(v @ inv) @ h
            ax = -(v / np.linalg.norm(v, axis=-1, keepdims=True)).sum(1)
            ax /= np.linalg.norm(ax, axis=-1, keepdims=True)
            u = d[:, j] / np.linalg.norm(d[:, j], axis=-1, keepdims=True)
            ang[j] = np.degrees(np.arccos(np.clip((u * ax).sum(-1), -1, 1))).mean()
            vc = v.mean(0)
            axc = -(vc / np.linalg.norm(vc, axis=-1, keepdims=True)).sum(0)
            angc[j] = np.degrees(np.arccos(np.clip(dc[j] @ axc / np.linalg.norm(dc[j]) / np.linalg.norm(axc), -1, 1)))
        # nearest heavy atom of another molecule to each H (centroids): C-H...X contacts
        heavy = np.where(self.z != 1)[0]
        dx = hc[:, None, :] - X[:, heavy].mean(0)[None]
        dx -= np.round(dx @ inv) @ h
        dist = np.linalg.norm(dx, axis=-1)
        dist[self.mol[self.hidx][:, None] == self.mol[heavy][None]] = np.inf
        return np.linalg.norm(dc, axis=-1), np.linalg.norm(d, axis=-1).mean(0), rg, ang, angc, dist.min(1)

    def frame(self, X, h, step):
        """X (P, n, 3) bead positions in A (unwrapped ring polymers), h (3,3) rows = lattice vectors (A)."""
        t0 = time.perf_counter()
        P = len(X)
        inv = np.linalg.inv(h)
        d = X - X[:1]  # i-PI wraps each bead separately when the forcefield has pbc: rebuild whole ring polymers
        X = X[:1] + d - np.round(d @ inv) @ h
        cell = np.ascontiguousarray(h, dtype=np.float32).reshape(9)
        if self.beads_mode == "centroid":
            Xs = X.mean(0, keepdims=True)
        elif self.beads_mode.isdigit():
            Xs = X[:: int(self.beads_mode)]
        else:
            Xs = X
        sig = self._sigma(Xs, cell)  # (B, M, n)
        self.timing["shield_s"] += time.perf_counter() - t0
        sig_bm = sig.mean(0)  # bead average (M, n)
        rec = {"step": step, "sigma": sig_bm[:, self.sel].T.astype(np.float32)}  # (nsel, M)
        # extras: per bead, the bead's own committee mean (or the contracted value for every bead)
        per_bead = sig if len(Xs) == P else np.repeat(sig_bm[None], P, 0)
        self.extras_list = []
        for b in range(P):
            e = {"shield_step": int(step), "sigma_iso": per_bead[b].mean(0)[self.sel].tolist(),
                 "sigma_iso_std": per_bead[b].std(0)[self.sel].tolist()}
            if self.extras_members:
                e["sigma_iso_members"] = per_bead[b][:, self.sel].tolist()
            self.extras_list.append(json.dumps(e))
        if self.est and step >= self.skip:  # the estimator only where it enters the statistics
            t1 = time.perf_counter()
            if self.units == "molecule":
                rec.update(self._estimator(X, h, cell, sig if len(Xs) == P else self._sigma(X, cell)))
            else:
                rec.update(self._estimator_atoms(X, h, cell))
            self.timing["estimator_s"] += time.perf_counter() - t1
        geo = self._geometry(X, h)
        rec.update(rCH_cent=geo[0].astype(np.float32), rCH_bead=geo[1].astype(np.float32),
                   rg_H=geo[2].astype(np.float32), bend=geo[3].astype(np.float32), bend_cent=geo[4].astype(np.float32),
                   dHX_min=geo[5].astype(np.float32))
        self._online(rec, X, h)
        if self.store and step >= self.skip:
            self.frames.append(rec)
        self.nsampled += 1
        if self.nsampled % 5 == 0:  # i-PI does not reliably run atexit handlers: keep the files current
            self.flush()

    def _estimator(self, X, h, cell, sig):
        """Scaled-coordinate FEP (H -> D for one molecule at a time). For molecule m with protio atoms H_m:
        q~_k = c + alpha (q_k - c) for its H beads, alpha = sqrt(m_H/m_D); the spring term is invariant, so
        w_m = exp(-beta/P sum_k [V(q~_k) - V(q_k)]) and <A>_D = <w A(q~)>_H / <w>_H. V differences per molecule
        from per-atom energies (delta model), each atom's change assigned to the nearest scaled molecule."""
        P = len(X)
        inv = np.linalg.inv(h)
        cen = X.mean(0)  # centroids (n,3)
        mc = []
        for a in self.mol_atoms:  # molecule centres, minimum image about the first atom
            d = cen[a] - cen[a[0]]
            mc.append(cen[a[0]] + (d - np.round(d @ inv) @ h).mean(0))
        mc = np.array(mc)
        e0 = self._eatoms(X, cell)  # (P, n)
        groups = self._groups(mc, h)
        nsel = len(self.sel)
        sel_mol = self.mol[self.sel]
        sigt = np.full((nsel, self.M), np.nan, np.float32)
        lnw = np.full(nsel, np.nan)
        lnw_mol = np.full(self.nmol, np.nan)
        for g in groups:
            hs = np.concatenate([self.mol_atoms[m][self.z[self.mol_atoms[m]] == 1] for m in g])
            Xt = X.copy()
            Xt[:, hs] = cen[hs] + self.alpha * (X[:, hs] - cen[hs])
            et = self._eatoms(Xt, cell)
            st = self._sigma(Xt, cell).mean(0)  # (M, n), bead average
            # nearest scaled molecule for every atom
            dv = cen[:, None, :] - mc[g][None]
            dv -= np.round(dv @ inv) @ h
            owner = np.array(g)[np.argmin(np.linalg.norm(dv, axis=-1), axis=1)]
            dE = (et - e0).sum(0)  # sum over beads (n,)
            for m in g:
                lnw_mol[m] = -self.beta / P * dE[owner == m].sum()
            for m in g:
                js = np.where(sel_mol == m)[0]
                sigt[js] = st[:, self.sel[js]].T
                lnw[js] = lnw_mol[m]
        return {"sigma_t": sigt, "lnw": lnw.astype(np.float64), "ngroups": len(groups)}

    def _estimator_atoms(self, X, h, cell):
        """Scaled-coordinate FEP per hydrogen-type atom (unit u: atom a_u, scale alpha_u = sqrt(m_from/m_to)). Units
        scaled together are >= dmin apart; each atom's energy change goes to the nearest scaled unit (H centroids)."""
        P = len(X)
        inv = np.linalg.inv(h)
        cen = X.mean(0)
        e0 = self._eatoms(X, cell)
        nu = len(self.est_atom)
        uc = cen[self.est_atom]
        groups, gc = [], []
        for u in self.rng.permutation(nu):
            for g, cg in zip(groups, gc):
                d = np.asarray(cg) - uc[u]
                d -= np.round(d @ inv) @ h
                if (np.linalg.norm(d, axis=1) >= self.dmin).all():
                    g.append(u); cg.append(uc[u]); break
            else:
                groups.append([u]); gc.append([uc[u]])
        sigt = np.full((nu, self.M), np.nan, np.float32)
        lnw = np.full(nu, np.nan)
        for g in groups:
            g = np.array(g)
            a, al = self.est_atom[g], self.est_alpha[g]
            Xt = X.copy()
            Xt[:, a] = cen[a] + al[None, :, None] * (X[:, a] - cen[a])
            et = self._eatoms(Xt, cell)
            st = self._sigma(Xt, cell).mean(0)
            dv = cen[:, None, :] - uc[g][None]
            dv -= np.round(dv @ inv) @ h
            owner = np.argmin(np.linalg.norm(dv, axis=-1), axis=1)
            dE = (et - e0).sum(0)
            for k, u in enumerate(g):
                lnw[u] = -self.beta / P * dE[owner == k].sum()
                if self.est_sel[u] >= 0:
                    sigt[u] = st[:, self.sel[self.est_sel[u]]]
        return {"sigma_t": sigt, "lnw": lnw, "ngroups": len(groups)}

    def _online(self, rec, X, h):
        if rec["step"] < self.skip:
            return
        s = rec["sigma"].astype(float)  # (nsel, M)
        self.fp_sigma.add([s.mean()])
        spread = s.std(1)
        self.spreads.append(spread.astype(np.float32))
        if self.acf is not None:
            self.acf.add(s.mean(1))
        if self.est:
            if not hasattr(self, "lnw_ref"):  # fixed reference per class (first frame's median) keeps sums finite
                self.lnw_ref = [float(np.nanmedian(np.where(self.est_class == c, rec["lnw"], np.nan)))
                                for c in range(self.nclass)]
            for c in range(self.nclass):
                ok = np.isfinite(rec["lnw"]) & (self.est_class == c) & (self.est_sel >= 0)
                lw = rec["lnw"][ok]
                w = np.exp(lw - self.lnw_ref[c])
                st, s0 = rec["sigma_t"][ok].astype(float), s[self.est_sel[ok]]
                self.jk[c].add(np.concatenate([(w[:, None] * st).sum(0), [w.sum(), (w * w).sum(), ok.sum(), lw.sum()],
                                               s0.sum(0), st.sum(0), (lw[:, None] * st).sum(0)]))
        self._flag(rec, X, h, spread)
        self._maybe_stop(rec["step"])

    def dsigma(self, tot, variant="fep"):
        """sigma_D - sigma_H per member + committee mean. fep: <w s~>/<w> - <s>; cumulant: second-order expansion
        <s~ - s> + cov(s~, ln w) (exact for Gaussian ln w; no exponential weights); scaling: <s~ - s> (bead spread
        only, centroid density unchanged: exact for harmonic modes)."""
        M = self.M
        A, B, W2, N, L = tot[:M], tot[M], tot[M + 1], tot[M + 2], tot[M + 3]
        S0, T, TL = tot[M + 4:2 * M + 4], tot[2 * M + 4:3 * M + 4], tot[3 * M + 4:4 * M + 4]
        if variant == "fep":
            d = A / B - S0 / N
        elif variant == "cumulant":
            d = (T - S0) / N + (TL / N - T / N * L / N)
        elif variant == "scaling":
            d = (T - S0) / N
        elif variant == "ess":
            return np.array([B * B / W2 / N])
        return np.concatenate([d, [d.mean()]])

    def _flag(self, rec, X, h, spread):
        hist = np.concatenate(self.spreads[-400:])
        if len(self.spreads) < 20:
            return
        thr = np.quantile(hist, 1 - self.uq_top)
        hits = np.where(spread > thr)[0]
        if not len(hits):
            return
        inv = np.linalg.inv(h)
        cen = X.mean(0)
        with open(self.out + "_flagged.xyz", "a") as f:
            for j in hits:
                a = self.sel[j]
                d = cen - cen[a]
                d -= np.round(d @ inv) @ h
                mols = np.unique(self.mol[np.linalg.norm(d, axis=1) < self.uq_radius])
                idx = np.concatenate([self.mol_atoms[m] for m in mols])
                pos = cen[a] + d[idx]
                f.write(f"{len(idx)}\nstep={rec['step']} atom={a} spread_ppm={spread[j]:.4f} "
                        f"thr_ppm={thr:.4f} sigma_members=\"{' '.join(f'{v:.3f}' for v in rec['sigma'][j])}\" "
                        f"centroid_cluster=T\n")
                for i, p in zip(idx, pos):
                    f.write(f"{chemical_symbols[self.z[i]]} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")
                self.nflag += 1

    def summary(self):
        s = {"sampled_frames": self.nsampled, "frames_in_stats": int(self.fp_sigma.n[0]),
             "sigma_mean_ppm": float(self.fp_sigma.mean[0]) if self.fp_sigma.n[0] else None,
             "sigma_se_ppm": float(self.fp_sigma.se()[0]) if self.fp_sigma.n[0] else None,
             "sigma_se_levels": self.fp_sigma.se_levels()[:, 0].tolist() if self.fp_sigma.n[0] else [],
             "flagged": self.nflag, "wall_s": time.time() - self.t0, **self.timing, "stopped": self.stopped}
        if self.est and self.jk[0].blocks:
            for c, jk in enumerate(self.jk):
                d = s if c == 0 else s.setdefault(f"class_{c}", {})  # class 0 keeps the single-class layout
                d.update(dsigma_blocks=len(jk.blocks), dsigma_block_len=jk.blen,
                         kish_ess_fraction=float(jk.estimate(lambda t: self.dsigma(t, "ess"))[0][0]))
                for v in ("fep", "cumulant", "scaling"):
                    est, se, k = jk.estimate(lambda t: self.dsigma(t, v))
                    d[v] = dict(dsigma_ppm=float(est[-1]), dsigma_se_ppm=float(se[-1]), ddelta_ppm=-float(est[-1]),
                                dsigma_members_ppm=est[:-1].tolist(), committee_std_ppm=float(np.std(est[:-1])))
        if self.acf is not None:
            s["acf"] = self.acf.result().tolist()
        return s

    def _maybe_stop(self, step):
        if self.target is None or self.stopped or step < self.skip + self.min_steps:
            return
        if self.stop_on in ("dsigma", "fep", "cumulant", "scaling") and self.est:
            _, se, k = self.jk[0].estimate(lambda t: self.dsigma(t, "fep" if self.stop_on == "dsigma" else self.stop_on))
            se = se[-1] if k >= 16 else np.nan
        else:
            se = self.fp_sigma.se()[0]
        if np.isfinite(se) and se < self.target:
            self.stopped = {"step": int(step), "se": float(se)}
            open("EXIT", "w").write(f"mlipcpp_ipi: SE {se:.4f} < target {self.target} ppm at step {step}\n")
            self.flush()

    def flush(self):
        with open(self.out + "_online.json", "w") as f:
            json.dump(self.summary(), f, indent=1)
        if self.store and self.frames:
            keys = list(self.frames[0])
            extra = {}
            if self.est:
                extra = dict(est_sel=self.est_sel, est_class=self.est_class, est_units=self.units)
                if self.units != "molecule":
                    extra.update(est_atom=self.est_atom, est_alpha=self.est_alpha)
            np.savez(self.out + "_frames.npz", **{k: np.array([r[k] for r in self.frames]) for k in keys},
                     sel=self.sel, mol=self.mol, hidx=self.hidx, hpar=self.hpar, z=self.z, **extra)


# ------------------------------------------------------------------------------------------------- i-PI driver
class MLIPCppDriver(Dummy_driver):
    def __init__(self, template, model, lib=None, backend="cuda", charge=0, spin=1, pbc=True, stress=True,
                 shield_models=None, shield_stride=1, shield_beads="all", shield_species="6", extras_members=False,
                 temperature=298, out="shield", store_frames=True, skip_steps=0, target_se=None, stop_on="dsigma",
                 min_steps=0, uq_top=0.01, uq_radius=6.0, acf_lags=0, estimator=None, delta_model=None,
                 est_dmin=9.0, est_molecules="all", m_from=1.00794, m_to=2.01410, seed=0, est_units="molecule",
                 *args, **kwargs):
        self.template, self.model_path = str(template), str(model)
        self.lib_path = lib or os.environ.get("MLIPCPP_C_LIB")
        if not self.lib_path:
            raise ValueError("give lib=/path/to/libmlipcpp_c.so or set MLIPCPP_C_LIB")
        self.backend_name = backend
        self.charge, self.spin = int(charge), int(spin)
        self.use_pbc, self.want_stress = _flag(pbc), _flag(stress)
        self.sh = dict(shield_models=shield_models, stride=int(shield_stride), beads=shield_beads,
                       species=[int(s) for s in str(shield_species).replace(";", ",").split(",") if s.strip()],
                       extras_members=_flag(extras_members), temperature=float(temperature), out=out,
                       store=_flag(store_frames), skip=int(skip_steps), target=target_se, stop_on=stop_on,
                       min_steps=int(min_steps), uq_top=float(uq_top), uq_radius=float(uq_radius),
                       acf_lags=int(acf_lags), estimator=estimator, delta_model=delta_model,
                       dmin=float(est_dmin), est_molecules=est_molecules, m_from=float(m_from), m_to=float(m_to),
                       seed=int(seed), est_units=est_units)
        super().__init__(*args, **kwargs)

    def check_parameters(self):
        import ase.io

        self.capi = CAPI(self.lib_path, self.backend_name)
        self.lib = self.capi.lib
        self.model = self.capi.load(self.model_path)
        at = ase.io.read(self.template)
        self.numbers = np.ascontiguousarray(at.numbers, dtype=np.int32)
        self.n = len(self.numbers)
        self.forces = np.zeros((self.n, 3), dtype=np.float32)
        self.stress6 = np.zeros(6, dtype=np.float32)
        self.pbc = np.array([self.use_pbc] * 3, dtype=np.bool_)
        self.popts = _PredictOptions(True, self.want_stress and self.use_pbc, False, self.charge, self.spin)
        self.ncall = 0
        self.engine = None
        if self.sh["shield_models"]:
            o = self.sh
            self.engine = ShieldingEngine(
                self.capi, self.numbers, at.positions, at.cell[:], _paths(o["shield_models"]), o["species"],
                o["beads"], o["temperature"], o["out"], o["store"], o["skip"], o["target"], o["stop_on"],
                o["min_steps"], o["uq_top"], o["uq_radius"], o["acf_lags"], o["estimator"], o["delta_model"],
                o["dmin"], o["est_molecules"], o["m_from"], o["m_to"], o["extras_members"], o["seed"],
                o["est_units"])

    def compute_structure(self, cell, pos):
        x = np.ascontiguousarray(unit_to_user("length", "angstrom", pos).reshape(-1, 3), dtype=np.float32)
        h = unit_to_user("length", "angstrom", np.asarray(cell))  # i-PI: columns are lattice vectors
        c = np.ascontiguousarray(h.T, dtype=np.float32).reshape(9)  # mlip.cpp: rows are lattice vectors
        res = self.capi.predict(self.model, self.numbers, x, c if self.use_pbc else None, self.pbc, self.popts)
        e = ctypes.c_float()
        self.lib.mlipcpp_result_get_energy(res, ctypes.byref(e))
        self.lib.mlipcpp_result_get_forces(res, self.forces.ctypes.data, self.n)
        vir = np.zeros((3, 3))
        if self.popts.compute_stress:
            self.lib.mlipcpp_result_get_stress(res, self.stress6.ctypes.data)
            xx, yy, zz, yz, xz, xy = self.stress6.astype(np.float64)  # Voigt, eV/A^3
            sigma = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
            vol = abs(np.linalg.det(h))
            vir = unit_to_internal("energy", "electronvolt", -sigma * vol)
        self.lib.mlipcpp_result_free(res)
        pot = unit_to_internal("energy", "electronvolt", float(e.value))
        frc = unit_to_internal("force", "ev/ang", self.forces.astype(np.float64)).reshape(np.shape(pos))
        return pot, frc, vir, ""

    def compute(self, cell, pos):
        if not isinstance(cell, list):
            if self.engine is not None and self.engine.est:
                raise ValueError("the shielding estimator needs ffdirect batch_size = number of beads")
            cell, pos, single = [cell], [pos], True
        else:
            single = False
        res = [self.compute_structure(c, p) for c, p in zip(cell, pos)]
        if self.engine is not None:
            step = self.ncall
            if step % self.sh["stride"] == 0:
                X = np.array([unit_to_user("length", "angstrom", p).reshape(-1, 3) for p in pos])
                h = unit_to_user("length", "angstrom", np.asarray(cell[0])).T
                self.engine.frame(X, h, step)
                self.extras = self.engine.extras_list
            ex = getattr(self, "extras", None)
            if ex is not None:
                res = [(r[0], r[1], r[2], ex[b] if b < len(ex) else ex[-1]) for b, r in enumerate(res)]
        self.ncall += 1
        return res[0] if single else res

    def __del__(self):
        if getattr(self, "engine", None) is not None:
            try:
                self.engine.flush()
            except Exception:
                pass
