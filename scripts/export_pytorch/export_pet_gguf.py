#!/usr/bin/env python3
"""
Export PET-MAD model to GGUF format with embedded computation graph.

Produces a single .gguf file containing:
1. Model weights as GGUF tensors
2. Computation graph as JSON string in metadata ("graph.json")
3. Model hyperparameters, species mappings, composition energies

Usage:
    uv run python3 scripts/export_pytorch/export_pet_gguf.py -o pet-auto.gguf
"""

import json
import argparse
import struct
import sys
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from export_pytorch.fx_converter import export_torch_model
from export_pytorch.export_pet_full import (
    PETFullModel, load_pet_model, get_model_params, load_local_model,
    get_species_mapping, get_composition_energies, get_energy_scale,
)

# GGUF format constants
GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3

# GGML tensor types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_I32 = 4

# GGUF value types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9


@dataclass
class GGUFTensor:
    """GGUF tensor descriptor."""
    name: str
    shape: List[int]
    dtype: int
    data: bytes
    offset: int = 0


class GGUFWriter:
    """Simple GGUF file writer."""

    def __init__(self):
        self.metadata: Dict[str, Tuple[int, Any]] = {}
        self.tensors: List[GGUFTensor] = []

    def add_string(self, key: str, value: str):
        self.metadata[key] = (GGUF_TYPE_STRING, value)

    def add_int32(self, key: str, value: int):
        self.metadata[key] = (GGUF_TYPE_INT32, value)

    def add_uint32(self, key: str, value: int):
        self.metadata[key] = (GGUF_TYPE_UINT32, value)

    def add_float32(self, key: str, value: float):
        self.metadata[key] = (GGUF_TYPE_FLOAT32, value)

    def add_array_int32(self, key: str, values: List[int]):
        self.metadata[key] = (GGUF_TYPE_ARRAY, (GGUF_TYPE_INT32, values))

    def add_array_float32(self, key: str, values: List[float]):
        self.metadata[key] = (GGUF_TYPE_ARRAY, (GGUF_TYPE_FLOAT32, values))

    def add_tensor(self, name: str, tensor: torch.Tensor, transpose_2d: bool = True):
        """Add a tensor to the GGUF file.

        Args:
            name: Tensor name
            tensor: PyTorch tensor
            transpose_2d: If True, transpose 2D weight matrices for GGML MUL_MAT
        """
        # Convert to float32 if needed
        if tensor.dtype in (torch.float16, torch.bfloat16):
            tensor = tensor.float()

        # Transpose 2D weight matrices for GGML MUL_MAT compatibility
        # GGML MUL_MAT: C = A @ B where A is [out, in] -> need [in, out] in GGML
        if transpose_2d and tensor.dim() == 2:
            tensor = tensor.T.contiguous()

        # Get shape in GGML format (reversed PyTorch shape)
        shape = list(tensor.shape)

        # Determine dtype
        if tensor.dtype == torch.float32:
            dtype = GGML_TYPE_F32
        elif tensor.dtype == torch.int32:
            dtype = GGML_TYPE_I32
        else:
            raise ValueError(f"Unsupported tensor dtype: {tensor.dtype}")

        # Convert to bytes
        data = tensor.detach().contiguous().numpy().tobytes()

        self.tensors.append(GGUFTensor(
            name=name,
            shape=shape,
            dtype=dtype,
            data=data,
        ))

    def write(self, path: str):
        """Write GGUF file."""
        with open(path, "wb") as f:
            # Write header
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self.tensors)))
            f.write(struct.pack("<Q", len(self.metadata)))

            # Write metadata
            for key, (vtype, value) in self.metadata.items():
                self._write_string(f, key)
                f.write(struct.pack("<I", vtype))

                if vtype == GGUF_TYPE_STRING:
                    self._write_string(f, value)
                elif vtype == GGUF_TYPE_INT32:
                    f.write(struct.pack("<i", value))
                elif vtype == GGUF_TYPE_UINT32:
                    f.write(struct.pack("<I", value))
                elif vtype == GGUF_TYPE_FLOAT32:
                    f.write(struct.pack("<f", value))
                elif vtype == GGUF_TYPE_ARRAY:
                    elem_type, arr = value
                    f.write(struct.pack("<I", elem_type))
                    f.write(struct.pack("<Q", len(arr)))
                    if elem_type == GGUF_TYPE_INT32:
                        for v in arr:
                            f.write(struct.pack("<i", v))
                    elif elem_type == GGUF_TYPE_FLOAT32:
                        for v in arr:
                            f.write(struct.pack("<f", v))

            # Write tensor info
            for tensor in self.tensors:
                self._write_string(f, tensor.name)
                f.write(struct.pack("<I", len(tensor.shape)))
                for dim in tensor.shape:
                    f.write(struct.pack("<Q", dim))
                f.write(struct.pack("<I", tensor.dtype))
                f.write(struct.pack("<Q", tensor.offset))

            # Align to 32 bytes
            current_pos = f.tell()
            alignment = 32
            padding = (alignment - (current_pos % alignment)) % alignment
            f.write(b"\x00" * padding)
            data_offset = f.tell()

            # Update tensor offsets and write data
            current_offset = 0
            for tensor in self.tensors:
                tensor.offset = current_offset
                current_offset += len(tensor.data)
                # Align each tensor to 32 bytes
                current_offset = (current_offset + 31) // 32 * 32

            # Rewrite tensor info with correct offsets
            f.seek(0)
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self.tensors)))
            f.write(struct.pack("<Q", len(self.metadata)))

            # Re-write metadata (same as before)
            for key, (vtype, value) in self.metadata.items():
                self._write_string(f, key)
                f.write(struct.pack("<I", vtype))
                if vtype == GGUF_TYPE_STRING:
                    self._write_string(f, value)
                elif vtype == GGUF_TYPE_INT32:
                    f.write(struct.pack("<i", value))
                elif vtype == GGUF_TYPE_UINT32:
                    f.write(struct.pack("<I", value))
                elif vtype == GGUF_TYPE_FLOAT32:
                    f.write(struct.pack("<f", value))
                elif vtype == GGUF_TYPE_ARRAY:
                    elem_type, arr = value
                    f.write(struct.pack("<I", elem_type))
                    f.write(struct.pack("<Q", len(arr)))
                    if elem_type == GGUF_TYPE_INT32:
                        for v in arr:
                            f.write(struct.pack("<i", v))
                    elif elem_type == GGUF_TYPE_FLOAT32:
                        for v in arr:
                            f.write(struct.pack("<f", v))

            # Re-write tensor info with updated offsets
            for tensor in self.tensors:
                self._write_string(f, tensor.name)
                f.write(struct.pack("<I", len(tensor.shape)))
                for dim in tensor.shape:
                    f.write(struct.pack("<Q", dim))
                f.write(struct.pack("<I", tensor.dtype))
                f.write(struct.pack("<Q", tensor.offset))

            # Seek to data section and write tensor data
            f.seek(data_offset)
            for tensor in self.tensors:
                f.write(tensor.data)
                # Pad to 32-byte alignment
                padding = (32 - (len(tensor.data) % 32)) % 32
                f.write(b"\x00" * padding)

    def _write_string(self, f, s: str):
        """Write a GGUF string (length-prefixed)."""
        encoded = s.encode("utf-8")
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)


def example_export_args(n_atoms: int, max_neighbors: int, cutoff: float, forces: bool = True):
    """Tracing inputs, their names and the torch.export dynamic shapes. Shared
    with scripts/train/compare_node_grads.py, which must export the exact same
    graph (same FX node names) to compare per-node gradients."""
    # Create test inputs for tracing
    torch.manual_seed(42)
    species = torch.zeros(n_atoms, dtype=torch.long)
    neighbor_species = torch.zeros(n_atoms, max_neighbors, dtype=torch.long)
    edge_vectors = torch.randn(n_atoms, max_neighbors, 3)
    padding_mask = torch.ones(n_atoms, max_neighbors, dtype=torch.bool)
    reverse_neighbor_index = torch.arange(n_atoms * max_neighbors, dtype=torch.long)

    if forces:
        cutoff_values_input = torch.full((n_atoms, max_neighbors), cutoff)
        example_inputs = (species, neighbor_species, edge_vectors,
                         padding_mask, reverse_neighbor_index, cutoff_values_input)
        input_names = ["species", "neighbor_species", "edge_vectors",
                       "padding_mask", "reverse_neighbor_index", "cutoff_values"]
    else:
        edge_distances = torch.rand(n_atoms, max_neighbors) * 3.0
        cutoff_factors = torch.ones(n_atoms, max_neighbors)
        example_inputs = (species, neighbor_species, edge_vectors, edge_distances,
                         padding_mask, reverse_neighbor_index, cutoff_factors)
        input_names = ["species", "neighbor_species", "edge_vectors", "edge_distances",
                       "padding_mask", "reverse_neighbor_index", "cutoff_factors"]

    # Use symbolic dynamic dimensions instead of numeric post-substitution.
    n_atoms_dim = torch.export.Dim("n_atoms", min=1, max=max(2, n_atoms))
    max_neighbors_dim = torch.export.Dim(
        "max_neighbors", min=1, max=max(2, max_neighbors)
    )
    n_edges_dim = torch.export.Dim(
        "n_edges", min=1, max=max(2, n_atoms * max_neighbors)
    )

    if forces:
        dynamic_shapes = (
            {0: n_atoms_dim},                      # species
            {0: n_atoms_dim, 1: max_neighbors_dim},  # neighbor_species
            {0: n_atoms_dim, 1: max_neighbors_dim},  # edge_vectors
            (                                       # *args
                {0: n_atoms_dim, 1: max_neighbors_dim},  # padding_mask
                {0: n_edges_dim},                       # reverse_neighbor_index
                {0: n_atoms_dim, 1: max_neighbors_dim},  # cutoff_values
            ),
        )
    else:
        dynamic_shapes = (
            {0: n_atoms_dim},                      # species
            {0: n_atoms_dim, 1: max_neighbors_dim},  # neighbor_species
            {0: n_atoms_dim, 1: max_neighbors_dim},  # edge_vectors
            (                                       # *args
                {0: n_atoms_dim, 1: max_neighbors_dim},  # edge_distances
                {0: n_atoms_dim, 1: max_neighbors_dim},  # padding_mask
                {0: n_edges_dim},                       # reverse_neighbor_index
                {0: n_atoms_dim, 1: max_neighbors_dim},  # cutoff_factors
            ),
        )


    return example_inputs, input_names, dynamic_shapes


def add_coulomb_input(inputs, names, shapes, n_atoms):
    """Append the geometry kernel; both axes share the atom dimension."""
    atom_dim = shapes[0][0]
    return (inputs + (torch.eye(n_atoms),), names + ["coulomb_matrix"],
            shapes[:3] + (shapes[3] + ({0: atom_dim, 1: atom_dim},),))


def add_conditioning_inputs(inputs, names, shapes, n_atoms):
    """Append the per-atom rows of the charge and spin tables (conditioned PETs)."""
    atom_dim = shapes[0][0]
    zeros = torch.zeros(n_atoms, dtype=torch.long)
    return (inputs + (zeros, zeros.clone()), names + ["charge_index", "spin_index"],
            shapes[:3] + (shapes[3] + ({0: atom_dim}, {0: atom_dim}),))


def state_dict_names(wrapper, pet):
    """GGUF tensor name -> metatrain state-dict name of the same parameter (by
    identity), for per-group learning rates (mlipcpp-train --lr-schedule)."""
    outer = getattr(pet, '_outer_model', [pet])[0]
    by_id = {}
    for name, p in outer.named_parameters():  # first (registered) name of shared ones
        by_id.setdefault(id(p), name)
    out = {n.replace(".", "_"): by_id[id(p)] for n, p in wrapper.named_parameters() if id(p) in by_id}
    if getattr(wrapper, "plato_lr", None) is not None:  # the folded mesh projections
        out.update(wrapper.plato.state_dict_names(by_id))
    return out


def metatrain_baselines(model):
    """Fitted composition (Z -> eV), energy scale and per-species
    non-conservative force scales of a metatrain >= 2026.5 PET (outer model,
    with `additive_models` and `scaler`). Empty dicts / None where absent."""
    # wrappers (LLPRUncertaintyModel in the upet checkpoints) keep that PET as .model
    while not hasattr(model, "additive_models") and isinstance(getattr(model, "model", None), torch.nn.Module):
        model = model.model

    def by_type(block):
        Z = block.samples.values[:, 0].tolist()
        return {int(z): float(v) for z, v in zip(Z, block.values.reshape(len(Z), -1)[:, 0])}

    comp, e_scale, f_scales = {}, None, {}
    for additive in getattr(model, "additive_models", []):
        weights = getattr(getattr(additive, "model", None), "weights", {})
        if "energy" in weights:
            comp = by_type(weights["energy"].block(0))
    scaler = getattr(getattr(model, "scaler", None), "model", None)
    scales = getattr(scaler, "scales", {})
    if "energy" in scales:
        e_scale = float(scales["energy"].block(0).values.reshape(-1)[0])
    if "non_conservative_force" in scales:
        # the scaler's per-type rows are labelled by type *index* (0..n_types-1),
        # in the order of scaler.atomic_types, not by atomic number
        block = scales["non_conservative_force"].block(0)
        Z = scaler.atomic_types.tolist()
        idx = block.samples.values[:, 0].tolist()
        vals = block.values.reshape(len(idx), -1)[:, 0]
        f_scales = {int(Z[i]): float(v) for i, v in zip(idx, vals)}
    return comp, e_scale, f_scales


def main():
    parser = argparse.ArgumentParser(
        description="Export PET model to GGUF with computation graph"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="pet-auto.gguf",
        help="Output GGUF file path",
    )
    parser.add_argument(
        "--model", type=str, default="pet-mad-1.0.2",
        help="Model name: 'pet-mad-1.0.2' (legacy), upet name like 'pet-mad-s', "
             "or a local metatrain checkpoint / make_init.py dump",
    )
    parser.add_argument(
        "--nc-forces", action="store_true",
        help="Also output the direct non-conservative force head: [n_atoms, 4] = (energy | force)",
    )
    parser.add_argument(
        "--no-forces", action="store_true",
        help="Export without forces support (energy only, smaller graph)",
    )
    parser.add_argument(
        "--n-atoms", type=int, default=7,
        help="Example atom count used for tracing",
    )
    parser.add_argument(
        "--max-neighbors", type=int, default=11,
        help="Example max-neighbors count used for tracing",
    )
    parser.add_argument(
        "--long-range-solver", choices=["direct-ewald", "p3m"], default=None,
        help="Required for PET-LR: p3m for scalable periodic inference, direct-ewald for reference",
    )
    args = parser.parse_args()

    n_atoms = args.n_atoms
    max_neighbors = args.max_neighbors

    print(f"Loading PET model: {args.model}...")
    pet = load_pet_model(args.model)
    pet.eval()

    params = get_model_params(pet)
    d_pet = params['d_pet']
    cutoff = params['cutoff']
    cutoff_width = params['cutoff_width']
    cutoff_function = params['cutoff_function']
    num_neighbors_adaptive = params['num_neighbors_adaptive']

    print(f"  d_pet={d_pet}, cutoff={cutoff}, cutoff_width={cutoff_width}")
    print(f"  cutoff_function={cutoff_function}, num_neighbors_adaptive={num_neighbors_adaptive}")
    print(f"  Export dimensions: n_atoms={n_atoms}, max_neighbors={max_neighbors}")
    print(f"  Forces mode: {not args.no_forces}")

    # Create wrapper with full computation path
    wrapper = PETFullModel(
        pet, n_atoms=n_atoms, max_neighbors=max_neighbors, d_pet=d_pet,
        forces=not args.no_forces, cutoff=cutoff, cutoff_width=cutoff_width,
        cutoff_function=cutoff_function, nc_forces=args.nc_forces,
    )
    wrapper.eval()
    if wrapper.long_range and args.long_range_solver is None:
        parser.error("PET-LR requires --long-range-solver p3m (scalable) or direct-ewald (reference)")
    if not wrapper.long_range and args.long_range_solver is not None:
        parser.error("--long-range-solver requires a checkpoint trained with long_range.enable=true")

    example_inputs, input_names, dynamic_shapes = example_export_args(
        n_atoms, max_neighbors, cutoff, forces=not args.no_forces)

    if wrapper.long_range:
        example_inputs, input_names, dynamic_shapes = add_coulomb_input(
            example_inputs, input_names, dynamic_shapes, n_atoms)
    if wrapper.conditioning:
        example_inputs, input_names, dynamic_shapes = add_conditioning_inputs(
            example_inputs, input_names, dynamic_shapes, n_atoms)

    # Export via torch.export
    print("\nExporting graph via torch.export...")
    graph, weights = export_torch_model(
        wrapper, example_inputs,
        output_path=None,
        input_names=input_names,
        input_dtypes={
            "species": "i32",
            "neighbor_species": "i32",
            "reverse_neighbor_index": "i32",
            "charge_index": "i32",
            "spin_index": "i32",
        },
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )
    print(f"  Graph: {len(graph.nodes)} nodes, {len(weights)} weights")

    # trainable tensors (mlipcpp-train): parameters only, never buffers or
    # lifted constants, and each exactly once
    named = list(wrapper.named_parameters(remove_duplicate=False))
    if len({id(p) for _, p in named}) != len(named):
        raise ValueError("shared parameters would be untied by mlipcpp-train")
    param_names = [n.replace(".", "_") for n, _ in named if n.replace(".", "_") in weights]

    graph_json = json.dumps(graph.to_dict())
    print(f"  Graph JSON: {len(graph_json)} bytes")

    # Get species mapping, composition energies, and energy scale
    species_to_index = get_species_mapping(pet)
    composition_energies = get_composition_energies(pet)
    energy_scale = get_energy_scale(pet)
    nc_force_scales = {}
    if Path(args.model).is_file():  # local metatrain model: its fitted baselines
        comp, e_scale, nc_force_scales = metatrain_baselines(load_local_model(args.model))
        composition_energies = comp or composition_energies
        energy_scale = e_scale if e_scale is not None else energy_scale

    print(f"  Species mapped: {len(species_to_index)}")
    print(f"  Composition energies: {len(composition_energies)}")
    print(f"  Energy scale: {energy_scale}")
    if energy_scale == 1.0:
        print("  Warning: energy_scale is 1.0 - verify this is correct for your model")

    # Validate composition energies
    composition_keys = list(composition_energies.keys())
    composition_values = list(composition_energies.values())
    assert len(composition_keys) == len(composition_values), \
        f"Composition keys ({len(composition_keys)}) and values ({len(composition_values)}) mismatch"

    # Write GGUF
    print(f"\nWriting GGUF to {args.output}...")
    writer = GGUFWriter()

    # General metadata
    writer.add_string("general.architecture", "pet-graph")
    writer.add_string("general.name", args.model)
    writer.add_string("general.version", "1.0.2")

    # Model hyperparameters
    writer.add_float32("pet.cutoff", cutoff)
    writer.add_float32("pet.cutoff_width", cutoff_width)
    writer.add_int32("pet.d_pet", d_pet)
    try:  # attention heads (GraphModel's packed-edge PET; older GGUFs: from the attention scale)
        writer.add_int32("pet.num_heads", int(pet.gnn_layers[0].trans.layers[0].attention.num_heads))
    except (AttributeError, IndexError):
        pass
    writer.add_float32("pet.energy_scale", energy_scale)
    writer.add_string("pet.cutoff_function", cutoff_function)
    writer.add_int32("pet.forces_mode", 1 if not args.no_forces else 0)
    writer.add_int32("pet.nc_forces", 1 if args.nc_forces else 0)
    if wrapper.long_range:
        calculator = pet.long_range_featurizer.ewald_calculator
        writer.add_string("pet.long_range.solver", args.long_range_solver)
        if args.long_range_solver == "p3m":
            calculator = pet.long_range_featurizer.p3m_calculator
            if calculator.mesh_interpolator.interpolation_nodes != 5:
                raise ValueError("Native P3M currently requires interpolation_nodes=5")
            writer.add_int32("pet.long_range.interpolation_nodes", 5)
        writer.add_float32("pet.long_range.smearing", float(calculator.potential.smearing))
        writer.add_float32("pet.long_range.wavelength", float(calculator.mesh_spacing if args.long_range_solver == "p3m" else calculator.lr_wavelength))
    if wrapper.conditioning:
        # charge in [-max_charge, max_charge], 2S+1 in [1, max_spin_multiplicity]
        writer.add_int32("pet.system_conditioning", 1)
        writer.add_int32("pet.max_charge", wrapper.max_charge)
        writer.add_int32("pet.max_spin_multiplicity", wrapper.max_spin_multiplicity)
    if wrapper.plato_lr is not None:
        from export_pytorch.plato_lr import gguf_metadata as plato_lr_metadata
        plato_lr_metadata(writer, wrapper.plato_lr)
    if nc_force_scales:
        writer.add_array_int32("pet.nc_force_scale_keys", list(nc_force_scales))
        writer.add_array_float32("pet.nc_force_scale_values", list(nc_force_scales.values()))
    writer.add_float32("pet.num_neighbors_adaptive",
                       float(num_neighbors_adaptive) if num_neighbors_adaptive is not None else 0.0)
    if params.get('cutoff_width_adaptive') is not None:
        writer.add_float32("pet.cutoff_width_adaptive", float(params['cutoff_width_adaptive']))
    if params.get('adaptive_cutoff_method') is not None:
        writer.add_string("pet.adaptive_cutoff_method", str(params['adaptive_cutoff_method']).lower())

    # Species mapping: pairs of [Z, index, Z, index, ...]
    species_map = []
    for Z, idx in sorted(species_to_index.items()):
        species_map.extend([Z, idx])
    writer.add_array_int32("pet.species_map", species_map)

    # Composition energies
    if composition_keys:
        writer.add_array_int32("pet.composition_keys", composition_keys)
        writer.add_array_float32("pet.composition_values", composition_values)

    # Computation graph as JSON string
    writer.add_string("graph.json", graph_json)
    writer.add_string("train.param_names", json.dumps(param_names))
    writer.add_string("train.state_dict_names", json.dumps(state_dict_names(wrapper, pet)))

    # Weight shapes as JSON (for loader to reconstruct tensors)
    weight_shapes = {name: list(t.shape) for name, t in weights.items()}
    writer.add_string("graph.weight_shapes", json.dumps(weight_shapes))

    # Add weight tensors (no transpose - graph handles layout)
    print(f"Adding {len(weights)} weight tensors...")
    for name, tensor in weights.items():
        writer.add_tensor(name, tensor, transpose_2d=False)

    writer.write(args.output)

    file_size = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"Done! {file_size:.1f} MB, {len(graph.nodes)} graph nodes, {len(weights)} weights")


if __name__ == "__main__":
    main()
