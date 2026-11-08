# i-PI with mlip.cpp forces and on-the-fly NMR shieldings

`mlipcpp_ipi.py` is an i-PI custom PES (`<ffdirect><pes>custom</pes><pes_path>mlipcpp_ipi.py</pes_path>`). It computes
forces with any mlip.cpp GGUF through `libmlipcpp_c.so`. Optionally it also evaluates a committee of per-atom models
(ShiftML4 exported with `scripts/export_pytorch/export_nanopet_gguf.py`, one GGUF per member) on the ring polymer.
On top of that it provides:

- **extras**: the per-bead committee mean and spread of σ_iso for the selected elements, returned as JSON
  extras (`sigma_iso`, `sigma_iso_std`, optionally `sigma_iso_members`), so `<trajectory extra_type="sigma_iso"> extras
  </trajectory>` writes them;
- **online statistics**: Flyvbjerg–Petersen blocking of the mean shielding and a block jackknife for the estimators
  (O(log N) and O(64) memory), written to `<out>_online.json`. When `target_se` is set, the PES writes `./EXIT` once
  the standard error falls below the target, and i-PI then performs a soft exit;
- **committee flags**: (frame, atom) pairs whose committee spread exceeds the running top-`uq_top` quantile. The
  centroid cluster around each flagged atom (radius `uq_radius`) is appended to `<out>_flagged.xyz` as a retraining
  candidate;
- **H→D estimator** (`estimator: H2D`): from an isotopically pure protio run, the shieldings each molecule would have
  if its H were D. It uses scaled coordinates: the H beads are contracted about their centroid by √(m_H/m_D), and the
  weight is w = exp(−β/P Σ_k ΔV_k). Molecules are scaled in groups at least `est_dmin` apart. ΔV per molecule comes
  from per-atom energies of a `make_delta_gguf.py` copy of the force model, and the observable is the scaled-bead
  shielding. Three variants are accumulated online: `fep` (exact reweighting), `cumulant` (second-order: no
  exponential weights) and `scaling` (bead contraction only, exact for harmonic modes). The weights assume a
  Boltzmann-sampled P-bead path integral (PILE, not PIGLET). `m_from`/`m_to` set the direction (e.g. 2.0141 → 1.00794
  expands the D beads of a deuterated run). `est_units: each` substitutes every H on its own; a list such as
  `1/1.00794/2.0141; 0/2.0141/1.00794` treats the k-th H on each carbon as its own class with its own masses, which
  gives both steps of CH₂Cl₂ → CHDCl₂ → CD₂Cl₂ from one CHDCl₂ run.
- `<out>_frames.npz`: reduced per-frame arrays, such as bead-averaged σ per member, σ̃, ln w and H geometry
  descriptors. These are a few kB per frame instead of the bead trajectories.

Shieldings need `<batch_size>P</batch_size>` in the ffdirect block, so that one driver call receives all P beads.
Shieldings are computed every `shield_stride` steps, on all beads, on the centroid (`shield_beads: centroid`) or on
every k-th bead. The full parameter list is in the module docstring.

```xml
<ffdirect name="f" pbc="true"><pes>custom</pes><pes_path>mlipcpp_ipi.py</pes_path><batch_size>16</batch_size>
  <parameters>{template: box.extxyz, model: pet-mols-s.gguf, lib: libmlipcpp_c.so, backend: cuda, temperature: 298,
    shield_models: shiftml4-*.gguf, shield_stride: 160, shield_species: 6, estimator: H2D,
    delta_model: pet-mols-s-delta.gguf, est_dmin: 7, skip_steps: 4000, target_se: 0.03, stop_on: cumulant}</parameters>
</ffdirect>
```

```
python make_delta_gguf.py pet-mols-s.gguf pet-mols-s-delta.gguf   # zero composition, forward only
```

Cost on an RTX PRO 6000 (625-atom CHCl₃ box): one force call takes about 11 ms per bead, one ShiftML4 member about
6 ms per bead (7 members: 42 ms), and one estimator frame takes P × (8 + 8 × n_groups) C-API calls (n_groups ≈ 8 at
7 Å).

Not supported: biasing on σ. That needs ∂σ/∂R, and the ShiftML graph has no backward pass in ggml. GELU (erf or
tanh) has no ggml backward op, and neither does the broadcast add of the centre embedding (ggml.c shape assert in
`ggml_build_backward_expand`). A bias would also need a weighted vector–Jacobian product, ∂(Σ c_i σ_i)/∂R, from the
C API.
