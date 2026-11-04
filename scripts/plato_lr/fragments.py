#!/usr/bin/env python3
"""Write <split>/fragment.bin for mlipcpp-train --frag-charge-weight: per atom (int32, the split's atom order) the
0-based index of its covalently bonded fragment within its structure. Two atoms are bonded when closer than
1.2 x (r_i + r_j), r the ASE covalent radii (ase.neighborlist.natural_cutoffs scaled by 1.2, as QCell's fragment check);
fragments are the connected components, numbered by their first atom. Open (non-periodic) structures only:
periodic ones get one fragment.

    python -I fragments.py SPLIT_DIR [SPLIT_DIR ...]
"""
import sys

import numpy as np
from ase.data import covalent_radii
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def main():
    for d in sys.argv[1:]:
        ns = int(np.load(f"{d}/ns.npy")); na = np.load(f"{d}/na.npy")
        x = np.fromfile(f"{d}/x.bin", np.float32).reshape(-1, 3); a = np.fromfile(f"{d}/a.bin", np.int32)
        c = np.fromfile(f"{d}/c.bin", np.float32).reshape(ns, 9)
        rc = 1.2 * covalent_radii[a]
        out = np.zeros(len(a), np.int32); counts = np.zeros(ns, np.int64)
        for s in range(ns):
            i0, i1 = na[s], na[s + 1]; n = i1 - i0
            if n < 2 or np.any(c[s] != 0):
                counts[s] = 1; continue
            p = cKDTree(x[i0:i1]).query_pairs(2 * rc[i0:i1].max(), output_type="ndarray")
            if len(p):
                dd = np.linalg.norm(x[i0 + p[:, 0]] - x[i0 + p[:, 1]], axis=1)
                p = p[dd < rc[i0 + p[:, 0]] + rc[i0 + p[:, 1]]]
            g = coo_matrix((np.ones(len(p)), (p[:, 0], p[:, 1])), shape=(n, n)) if len(p) else coo_matrix((n, n))
            k, lab = connected_components(g, directed=False)
            _, first = np.unique(lab, return_index=True)          # renumber by first atom
            order = np.argsort(np.argsort(first))
            out[i0:i1] = order[lab]; counts[s] = k
        out.tofile(f"{d}/fragment.bin")
        h = np.bincount(np.minimum(counts, 10))
        print(f"{d}: {ns} structures, {len(a)} atoms; fragments per structure 1..10+: {h[1:].tolist()}")


if __name__ == "__main__":
    main()
