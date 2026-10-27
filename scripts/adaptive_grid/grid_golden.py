#!/usr/bin/env python3
"""Golden values of metatrain's "grid" adaptive cutoff (get_adaptive_cutoffs_grid, float64,
autograd for d r_i / d d_ik) for tests/test_adaptive_grid.cpp.

    PYTHONPATH=<metatrain src> python grid_golden.py tests/resources/adaptive_grid_golden.json
"""
import json
import sys

import numpy as np
import torch
from metatrain.pet.modules.adaptive_cutoff import get_adaptive_cutoffs_grid

cases = []
rng = np.random.default_rng(0)
for cutoff, width, target, counts in ((8.0, 0.5, 16.0, (40, 5, 120, 17)), (4.5, 1.0, 8.0, (12, 3, 30))):
    centers, dist = [], []
    for i, n in enumerate(counts):
        # a dense shell, a sparse one, uniform: radii on both sides of the probes
        d = np.sort(np.concatenate([rng.uniform(0.9, 2.0, n // 3), rng.uniform(2.0, cutoff, n - n // 3)]))
        centers += [i] * n
        dist += list(d)
    c = torch.tensor(centers)
    d = torch.tensor(dist, dtype=torch.float64, requires_grad=True)
    r = get_adaptive_cutoffs_grid(c, d, target, len(counts), cutoff, cutoff_width=width)
    jac = np.zeros(len(dist))
    for i in range(len(counts)):
        (g,) = torch.autograd.grad(r[i], d, retain_graph=True)
        jac += g.numpy()  # each edge moves only its centre's radius
    cases.append(dict(cutoff=cutoff, width=width, target=target, centers=centers, distances=dist,
                      radii=r.detach().tolist(), jacobian=jac.tolist()))
json.dump(cases, open(sys.argv[1], "w"), indent=1)
