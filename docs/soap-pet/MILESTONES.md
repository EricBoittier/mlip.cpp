# SOAP-PET / LODE milestones (PET-OMol-S fine-tuning, Plato-PET-LR)

Shared bases:
- mlip.cpp: `feat/pet-soap-base` (this branch). It is feat/pet-plato-lr d2e4369 plus the generic GIR/GGUF fixes from
  feat/pet-soap-v4 9e117ce: SELECT innermost axis, logical_not, index_select dim 1, I32 type code and buffers,
  two-way broadcast, REPEAT template, 1-D SELECT, and the debug env vars MLIPCPP_TRACE_NODES / MLIPCPP_DUMP_DIR.
- metatrain: `feat/pet-soap` (SoapPowerSpectrum core; contract in /home/boittier/data/soap-pet/INTERFACE.md).

Branches per arm: core + v3 (additive SOAP head), v1 (node token), v2 (edge tokens), v4 (attention bias),
v5c (cross-attention to SOAP environments within ~8 A), LODE (+ (d) reciprocal-space cross-attention).
Every arm sits behind its own zero-init gate.

## M1: synchronised and exact (gate)
Each arm, before any training job:
1. Its mlip.cpp branch is rebased onto `feat/pet-soap-base`, and its metatrain branch onto the current `feat/pet-soap` head.
2. SOAP on PET's adaptive neighbours is cutoff-weighted (the core's `weight=` argument, PET cutoff_factors), so no
   energy jumps occur when a neighbour crosses its pair cutoff.
3. The GGUF is built from Fa's input (`agents/petlr/F/in/Fa.gguf`: E-init, les_tail_sigma 4, LES gate 0.2). If rebuilt via
   gguf_to_checkpoint, run `les_tail_variant.py --tail-sigma 4` first, otherwise the base energies are off by 0.2-0.4 eV/atom.
4. Parity:
   - Gate 0 matches the base GGUF to within 1e-6 eV/atom.
   - Gate 1: torch vs mlip.cpp within 1e-5 eV/atom and 1e-4 eV/A on CPU (parity_pet_plato_lr.py), plus one GPU check.
5. The new gate and its weights are matched by the lr-schedule groups (the train.log "group X: N tensors" lines; patterns
   use train.state_dict_names).

## M2: signal (one ~15 min job per arm, clariden ab010)
For each arm, record in /home/boittier/data/soap-pet/STATUS.md: the fitted gate or the gate after the short fine-tune,
the train/val loss against an Fa-setup control over the same steps, the per-step overhead, and, where relevant, QCell
charged+neutral 4-8 A and ion-pair numbers (scripts/lesq.py, fragq.py templates).

## M3: proposal
A ranked list of arms for 4-seed runs (Fa setup, seeds 13/21/42/63, about 5 h per arm on one node) with the cost.
No multi-seed jobs are submitted without the user's approval.
