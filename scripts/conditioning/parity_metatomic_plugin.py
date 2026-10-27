#!/usr/bin/env python3
"""Charge / spin through the mlipcpp metatomic plugin (metatomic-core, the C API engines
use): requested inputs, the systems' "charge" / "spin_multiplicity" custom data, and the
energies / forces against mlipcpp-eval (the C++ API) on the same frames.

    python parity_metatomic_plugin.py PLUGIN.so MODEL.gguf FRAMES.xyz MLIPCPP_EVAL.json

FRAMES.xyz / MLIPCPP_EVAL.json: parity_pet.py's OUT/frames.xyz and mlipcpp-eval's output on
it. Needs the metatomic_core and metatensor Python packages of the metatomic version the
plugin was built against.
"""
import json
import sys

import ase.io
import metatensor
import numpy as np

import metatomic as mta


def scalar(value, name):
    block = metatensor.TensorBlock(
        values=np.array([[float(value)]], dtype=np.float64),
        samples=metatensor.Labels(["system"], np.array([[0]], dtype=np.int32)),
        components=[], properties=metatensor.Labels([name], np.array([[0]], dtype=np.int32)))
    return metatensor.TensorMap(metatensor.Labels.single(), [block])


def system(atoms, q=None, s=None):
    cell = np.array(atoms.cell[:], dtype=np.float64)
    cell[~atoms.pbc] = 0.0
    sy = mta.System("angstrom", atoms.numbers.astype(np.int32), atoms.positions.astype(np.float64), cell,
                    atoms.pbc.astype(bool))
    if q is not None:
        sy.add_custom_data("charge", scalar(q, "charge"))
    if s is not None:
        sy.add_custom_data("spin_multiplicity", scalar(s, "spin_multiplicity"))
    return sy


def main():
    plugin, gguf, xyz, ref = sys.argv[1:5]
    mta.load_plugin(plugin)
    metatensor.register_external_data_wrapper("metatensor::SimpleDataArray", metatensor.ExternalCpuArray)
    model = mta.load_model(gguf, {"dtype": "float64"})
    energy = mta.Quantity(name="energy", unit="eV", sample_kind="system", gradients=["positions"])
    inputs = model.requested_inputs([energy])
    print("requested inputs:", [(q.name, q.sample_kind) for q in inputs])
    print("interaction range:", model.capabilities().interaction_range)
    frames = ase.io.read(xyz, ":")
    got = json.load(open(ref))["frames"]
    worst_e = worst_f = 0.0
    n = 0
    for at, g in zip(frames, got):
        if "error" in g:
            q, s = int(at.info["charge"]), int(at.info["spin_multiplicity"])
            try:
                mta.execute_model(model, [system(at, q, s)], [energy], check_consistency=True)
                print(f"q={q} 2S+1={s}: NO error through the plugin")
                return 1
            except Exception as e:  # noqa: BLE001
                print(f"q={q} 2S+1={s}: error through the plugin too ({str(e).splitlines()[0][:90]})")
            continue
        (e,) = mta.execute_model(model, [system(at, g["charge"], g["spin_multiplicity"])], [energy],
                                 check_consistency=True)
        b = e.block()
        E = float(b.values[0, 0])
        F = -np.asarray(b.gradient("positions").values)[:, :, 0]
        worst_e = max(worst_e, abs(E - g["energy"]))
        worst_f = max(worst_f, float(np.abs(F - np.array(g["forces"]).reshape(-1, 3)).max()))
        n += 1
    # without the data: metatomic itself refuses (requested inputs are required with
    # check_consistency); without the check the plugin uses the defaults (neutral singlet)
    try:
        mta.execute_model(model, [system(frames[0])], [energy], check_consistency=True)
        print("no data, check_consistency: accepted")
    except Exception as exc:  # noqa: BLE001
        print("no data, check_consistency: metatomic refuses:", str(exc).splitlines()[0][:110])
    (e0,) = mta.execute_model(model, [system(frames[0])], [energy], check_consistency=False)
    d0 = abs(float(e0.block().values[0, 0]) - got[0]["energy"]) if (got[0]["charge"], got[0]["spin_multiplicity"]) == (0, 1) else float("nan")
    print(f"{n} frames: plugin vs mlipcpp-eval max |dE| {worst_e:.2e} eV, max |dF| {worst_f:.2e} eV/A; "
          f"no data = (0, 1): |dE| {d0:.2e}")
    ok = worst_e < 1e-6 and worst_f < 1e-5 and d0 < 1e-6
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
