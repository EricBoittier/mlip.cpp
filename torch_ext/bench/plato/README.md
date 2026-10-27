# Plato-PET-LR with the fused attention

metatrain `experimental.plato_pet_lr`, `attention_backend: fused` (EricBoittier/metatrain
`feat/plato-fused-attn`) uses `torch.ops.mlipcpp.pet_attention` unchanged: the `F` frames
are folded into `F x num_heads` heads that share Plato's per-key `log f_c` bias, and the
frame average of attention-only frames is the `w_f` average of those heads' outputs.

Scripts (need that metatrain on `PYTHONPATH`, `PLATO_TEST_XYZ` = the bio_dimers test set):

- `parity.py OUT name=CKPT ...`: fused vs manual, E / F / stress and E + F loss parameter
  gradients, float64 and float32, open dimers, periodic dimers, the 160-atom cell.
- `bench_fused.py CKPT OUT backend ...`: E + F calls and eager train steps, time and peak memory.
- `export_check.py save|load`: metatomic export with the op, reload in a fresh process.
- `profile_step.py`: CUDA kernel time by group (attention vs the rest).

Results in `../../results/plato/` (`*_rtx4070tis*`: cosmopc27, torch 2.13; `*_gh200*`: CSCS
daint GH200, torch 2.14.1).
