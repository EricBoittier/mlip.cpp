#!/usr/bin/env python3
"""Dump the freshly initialised PET that `mtt train <options>` would train.

Runs metatrain's own CLI with the PET trainer's `train` replaced by a dump, so
seeding, atomic-type inference and target registration follow exactly the
path of the reference run. Run it from the directory holding train/ val/ test/.

    python make_init.py options.yaml -o init.pt
"""
import argparse
import sys

import torch

import metatrain.pet.trainer as pet_trainer
from metatrain.__main__ import main as mtt_main


class _Done(BaseException):  # mtt catches Exception and exits 1
    pass


def dump(path):
    def train(self, model, dtype, devices, train_datasets, val_datasets, checkpoint_dir):
        torch.save({"model": model, "hypers": self.hypers}, path)
        print(f"saved initial PET ({sum(p.numel() for p in model.parameters())} params) to {path}")
        raise _Done
    return train


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("options")
    ap.add_argument("-o", "--output", default="init.pt")
    args = ap.parse_args()
    pet_trainer.Trainer.train = dump(args.output)
    sys.argv = ["mtt", "train", args.options, "-o", "unused.pt"]
    try:
        mtt_main()
    except _Done:
        pass
