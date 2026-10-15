"""Generate small, untrained PET-LR parity fixtures using metatrain + torch-pme.

python tests/generate_pet_lr_reference.py /tmp/pet-lr-reference
MLIPCPP_PET_LR_REFERENCE=/tmp/pet-lr-reference/reference.json build/tests/test_coulomb

Selects upstream Ewald or P3M through the training-path solver setting, avoiding
unfitted composition baselines. No downloaded pretrained weights needed.
"""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import torch
from metatomic.torch import ModelOutput, System
from metatrain.pet import PET
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--solver", choices=["direct-ewald", "p3m"], default="direct-ewald"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    info = DatasetInfo(
        length_unit="angstrom",
        atomic_types=[1, 8],
        targets={
            "energy": get_energy_target_info(
                "energy", {"quantity": "energy", "unit": "eV"}
            )
        },
    )
    cases = []
    for kind, enabled in (
        ("feedforward", True),
        ("residual", True),
        ("residual", False),
    ):
        torch.manual_seed(123)
        hypers = copy.deepcopy(get_default_hypers("pet")["model"])
        hypers.update(
            d_pet=16,
            d_node=16,
            d_head=16,
            d_feedforward=32,
            num_heads=2,
            num_attention_layers=1,
            num_gnn_layers=2,
            featurizer_type=kind,
            num_neighbors_adaptive=None,
            cutoff=4.5,
            cutoff_width=0.5,
            cutoff_function="Cosine",
            normalization="LayerNorm",
            activation="SiLU",
        )
        hypers["long_range"].update(
            enable=enabled,
            use_ewald=args.solver == "direct-ewald",
            smearing=1.4,
            kspace_resolution=1.33,
        )
        model = PET(hypers, info)
        label = kind if enabled else f"{kind}-sr"
        ckpt = args.output / f"{label}.pt"
        torch.save({"model": model}, ckpt)
        gguf = args.output / f"{label}.gguf"
        subprocess.run(
            [
                sys.executable,
                "scripts/export_pytorch/export_pet_gguf.py",
                "--model",
                str(ckpt),
                "-o",
                str(gguf),
                *(["--long-range-solver", args.solver] if enabled else []),
            ],
            check=True,
        )
        # Training mode excludes unfitted baselines; use_ewald selects the solver.
        model.train()
        for name, points, cell in [
            ("molecule", [[0.3, 0.5, 0.2], [1.5, 0.8, 0.4], [5.1, 1.7, 2.1]], None),
            ("isolated", [[0.0, 0.0, 0.0]], None),
            ("distant", [[0.0, 0.0, 0.0], [6.0, 1.0, 0.0]], None),
            (
                "triclinic",
                [[0.3, 0.5, 0.2], [1.5, 0.8, 0.4], [5.1, 1.7, 2.1]],
                [[7.3, 0, 0], [0.6, 6.1, 0], [0.2, 0.7, 5.7]],
            ),
            (
                "images",
                [[0.3, 0.5, 0.2], [1.5, 0.8, 0.4]],
                [[3.3, 0, 0], [0.6, 3.1, 0], [0.2, 0.7, 3.7]],
            ),
        ] + (
            [
                (
                    "periodic-729",
                    [
                        [2.1 * x + 0.13, 2.1 * y + 0.23, 2.1 * z + 0.31]
                        for x in range(9)
                        for y in range(9)
                        for z in range(9)
                    ],
                    [[18.9, 0, 0], [0, 18.9, 0], [0, 0, 18.9]],
                )
            ]
            if args.solver == "p3m"
            else []
        ):
            r = torch.tensor(points, requires_grad=True)
            c = torch.tensor(cell if cell else [[0.0] * 3] * 3)
            strain = torch.eye(3, requires_grad=True)
            types = torch.tensor(([8] + [1] * (len(points) - 1)), dtype=torch.int32)
            system = System(
                types=types,
                positions=r @ strain,
                cell=c @ strain,
                pbc=torch.tensor([cell is not None] * 3),
            )
            system = get_system_with_neighbor_lists(
                system, model.requested_neighbor_lists()
            )
            energy = (
                model([system], {"energy": ModelOutput(sample_kind="system")})["energy"]
                .block()
                .values.sum()
            )
            grad, ds = torch.autograd.grad(energy, (r, strain))
            # Independent torch-pme matrix (one unit charge channel per atom).
            from metatrain.utils.long_range import LongRangeFeaturizer

            lr = (
                model.long_range_featurizer
                if enabled
                else LongRangeFeaturizer(
                    hypers["long_range"], 16, model.requested_neighbor_lists()[0]
                )
            )
            if cell:
                nl = system.get_neighbor_list(model.requested_neighbor_lists()[0])
                indices = nl.samples.values[:, :2]
                distances = nl.values.reshape(-1, 3).norm(dim=1)
                calc = (
                    lr.p3m_calculator if args.solver == "p3m" else lr.ewald_calculator
                )
            else:
                indices = torch.combinations(torch.arange(len(points)), 2)
                distances = (r[indices[:, 1]] - r[indices[:, 0]]).norm(dim=1)
                calc = lr.direct_calculator
            matrix = (
                calc(torch.eye(len(points)), c, r, indices, distances)
                if len(points) <= 512
                else None
            )
            case = {
                "name": f"{label}-{name}",
                "solver": args.solver,
                "gguf": str(gguf.resolve()),
                "positions": r.detach().flatten().tolist(),
                "numbers": types.tolist(),
                "energy": energy.item(),
                "forces": (-grad).flatten().tolist(),
            }
            if matrix is not None:
                case["matrix"] = matrix.detach().flatten().tolist()
            if cell:
                stress = (ds + ds.T) * 0.5 / torch.det(c).abs()
                case.update(
                    cell=cell,
                    stress=stress[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]].tolist(),
                )
            cases.append(case)
    energy_gguf = args.output / "energy-only.gguf"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_pytorch/export_pet_gguf.py",
            "--model",
            str(args.output / "feedforward.pt"),
            "--long-range-solver",
            args.solver,
            "--no-forces",
            "-o",
            str(energy_gguf),
        ],
        check=True,
    )
    cases.extend(
        [
            dict(
                case,
                name=case["name"] + "-energy-only",
                gguf=str(energy_gguf.resolve()),
                energy_only=True,
            )
            for case in cases
            if case["name"].startswith("feedforward-")
        ]
    )
    (args.output / "reference.json").write_text(json.dumps(cases, indent=2))
    print(f"Wrote {len(cases)} reference cases to {args.output / 'reference.json'}")


if __name__ == "__main__":
    main()
