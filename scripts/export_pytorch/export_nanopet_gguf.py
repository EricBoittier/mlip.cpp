#!/usr/bin/env python3
"""Export a metatrain nanoPET per-atom scalar model (e.g. one ShiftML4 committee
member) to a "pet-graph" GGUF that GraphModel runs.

The exported graph is the nanoPET forward on GraphModel's padded inputs
(species, neighbor_species, edge_vectors, padding_mask, reverse_neighbor_index,
cutoff_values). Its per-atom output y_i is the selected scalar target block,
pre-multiplied by --output-factor; GraphModel then reports

    value_i = pet.energy_scale * y_i + composition[Z_i]

per atom (mlipcpp_result_get_atomic_values), with energy_scale the metatrain
Scaler factor and composition the CompositionModel weights of that block, both
times --output-factor. The "energy" of the result is the sum over atoms.

ShiftML4 (``mtt::cs_iso``) predicts the shielding tensor as spherical blocks
lambda = 0, 1, 2 from independent linear heads on the same invariant node
features; the isotropic shielding is trace / 3 = -(lambda=0 block) / sqrt(3), so
the default exports the lambda = 0 block with output factor -1/sqrt(3) and the
per-atom values are sigma_iso in ppm.

Usage (needs torch + metatomic-torch to open the TorchScript archive):
    python export_nanopet_gguf.py ~/.cache/shiftml/ShiftML41/ShiftML41.pt -o shiftml4-1.gguf
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from export_pytorch.fx_converter import export_torch_model  # noqa: E402
from export_pytorch.export_pet_gguf import GGUFWriter, example_export_args  # noqa: E402


def _layer(sd, prefix):
    return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}


class Attention(torch.nn.Module):
    def __init__(self, sd, d, heads):
        super().__init__()
        self.heads = heads
        self.ln = torch.nn.LayerNorm(d)
        self.inp = torch.nn.Linear(d, 3 * d, bias=False)
        self.out = torch.nn.Linear(d, d, bias=False)
        self.ln.weight.data, self.ln.bias.data = sd["layernorm.weight"], sd["layernorm.bias"]
        self.inp.weight.data = sd["in_proj.weight"]
        self.out.weight.data = sd["out_proj.weight"]

    def forward(self, x, mask):  # x [n, M, d], mask [n, M]
        n, m, d = x.shape
        h = self.heads
        q, k, v = torch.chunk(self.inp(self.ln(x)), 3, dim=-1)
        q = q.reshape(n, m, h, d // h).transpose(1, 2)
        k = k.reshape(n, m, h, d // h).transpose(1, 2)
        v = v.reshape(n, m, h, d // h).transpose(1, 2)
        a = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d // h), dim=-1)
        # nanoPET: radial weights on the keys after the softmax, renormalised
        a = a * mask.unsqueeze(1).unsqueeze(2)
        a = a / (a.sum(dim=-1, keepdim=True) + 1e-6)
        o = torch.matmul(a, v).transpose(1, 2).reshape(n, m, d)
        return (self.out(o) + x) * 0.7071067811865476


class FeedForward(torch.nn.Module):
    def __init__(self, sd, d, dff):
        super().__init__()
        self.ln = torch.nn.LayerNorm(d)
        self.mlp = torch.nn.Linear(d, dff, bias=False)
        self.out = torch.nn.Linear(dff, d, bias=False)
        self.ln.weight.data, self.ln.bias.data = sd["layernorm.weight"], sd["layernorm.bias"]
        self.mlp.weight.data = sd["mlp.weight"]
        self.out.weight.data = sd["output.weight"]

    def forward(self, x):
        return (self.out(torch.nn.functional.gelu(self.mlp(self.ln(x)))) + x) * 0.7071067811865476


class Transformer(torch.nn.Module):
    def __init__(self, sd, d, dff, heads):
        super().__init__()
        n_layers = len({k.split(".")[1] for k in sd if k.startswith("layers.")})
        self.att = torch.nn.ModuleList(
            [Attention(_layer(sd, f"layers.{i}.attention_block."), d, heads) for i in range(n_layers)])
        self.ff = torch.nn.ModuleList(
            [FeedForward(_layer(sd, f"layers.{i}.ff_block."), d, dff) for i in range(n_layers)])

    def forward(self, x, mask):
        for att, ff in zip(self.att, self.ff):
            x = ff(att(x, mask))
        return x


class NanoPETGraph(torch.nn.Module):
    """nanoPET (metatrain.experimental.nanopet) on GraphModel's padded inputs."""

    def __init__(self, net, block: str, factor: float):
        super().__init__()
        sd = {k: v.detach().clone().float() for k, v in net.state_dict().items()}
        self.cutoff = float(net.cutoff)
        self.width = float(net.cutoff_width)
        heads = int(getattr(net.transformer.layers, "0").attention_block.num_heads)
        d = sd["encoder.compressor.weight"].shape[0]
        dff = sd["transformer.layers.0.ff_block.mlp.weight"].shape[0]
        self.d = d
        if int(net.num_mp_layers) != 1:
            raise NotImplementedError("exporter written for num_mp_layers == 1 (one GNN contraction)")

        enc = _layer(sd, "encoder.")
        dims = [enc["cartesian_encoder.0.weight"], enc["cartesian_encoder.2.weight"],
                enc["cartesian_encoder.4.weight"]]
        self.cart = torch.nn.ModuleList()
        for w in dims:
            lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=False)
            lin.weight.data = w
            self.cart.append(lin)
        self.center = torch.nn.Embedding(*enc["center_encoder.weight"].shape)
        self.center.weight.data = enc["center_encoder.weight"]
        self.neighbor = torch.nn.Embedding(*enc["neighbor_encoder.weight"].shape)
        self.neighbor.weight.data = enc["neighbor_encoder.weight"]
        # compressor(cat[cartesian, center, neighbor]) as three blocks, so that the
        # per-atom centre term broadcasts over the neighbour slots (no expand)
        wc = enc["compressor.weight"]
        self.comp_cart, self.comp_center, self.comp_neighbor = (
            torch.nn.Linear(d, d, bias=False) for _ in range(3))
        self.comp_cart.weight.data = wc[:, :d].contiguous()
        self.comp_center.weight.data = wc[:, d:2 * d].contiguous()
        self.comp_neighbor.weight.data = wc[:, 2 * d:].contiguous()

        self.transformer = Transformer(_layer(sd, "transformer."), d, dff, heads)
        self.contraction = torch.nn.Linear(2 * d, d, bias=False)
        self.contraction.weight.data = sd["gnn_contractions.0.weight"]
        self.gnn_transformer = Transformer(_layer(sd, "gnn_transformers.0."), d, dff, heads)

        (target,) = [k for k in net.head_types]
        if net.head_types[target] != "linear":
            raise NotImplementedError("only 'linear' (identity) heads are supported")
        w = sd[f"last_layers.{target}.{block}.weight"]
        if w.shape[0] != 1:
            raise NotImplementedError(f"{block} has {w.shape[0]} components; export a scalar block")
        self.last = torch.nn.Linear(d, 1, bias=False)
        self.last.weight.data = w * factor

    def forward(self, species, neighbor_species, edge_vectors, padding_mask,
                reverse_neighbor_index, cutoff_values):
        # GraphModel forces-mode inputs (export_pet_gguf.example_export_args)
        n = species.shape[0]
        m = neighbor_species.shape[1]
        d = self.d
        r = torch.sqrt((edge_vectors * edge_vectors).sum(dim=-1))
        x = torch.clamp((r - (cutoff_values - self.width)) / self.width, 0.0, 1.0)
        f = 0.5 * (1.0 + torch.cos(math.pi * x))
        mask = torch.where(padding_mask, torch.zeros_like(f), f)

        c = edge_vectors
        c = torch.nn.functional.silu(self.cart[0](c))
        c = torch.nn.functional.silu(self.cart[1](c))
        c = self.cart[2](c)
        ce = self.comp_center(self.center(species)).unsqueeze(1)
        ne = self.neighbor(neighbor_species.flatten()).view(n, m, d)
        feats = self.comp_cart(c) + self.comp_neighbor(ne) + ce
        feats = self.transformer(feats, mask)

        flat = feats.reshape(n * m, d)
        rev = flat[reverse_neighbor_index]
        new = self.contraction(torch.cat([flat, rev], dim=-1)).view(n, m, d)
        new = self.gnn_transformer(new, mask)
        feats = (feats + new) * 0.7071067811865476

        node = (feats * mask.unsqueeze(-1)).sum(dim=1)
        return self.last(node).squeeze(-1)


def load_scripted(path):
    import metatomic.torch  # noqa: F401  (registers the TorchScript classes)
    m = torch.jit.load(path, map_location="cpu")
    return m, m.module


def baselines(net, block_index: int, factor: float):
    """Scaler factor and per-Z composition weights of the block (times factor)."""
    target = list(net.head_types)[0]
    scale = float(net.scaler.scales[net.scaler.output_name_to_output_index[target]])
    am = getattr(net.additive_models, "0")
    comp = {}
    if target in am.outputs:
        w = am.weights[target].block(block_index).values.reshape(len(am.atomic_types), -1)
        if w.shape[1] != 1:
            raise NotImplementedError("composition block is not scalar")
        comp = {int(z): float(w[i, 0]) * factor for i, z in enumerate(am.atomic_types)}
    return scale, comp


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("model", help="TorchScript metatomic archive with a nanoPET network")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--block", default="mtt::cs_iso_o3_lambda_0_o3_sigma_1")
    p.add_argument("--block-index", type=int, default=0, help="block position in the target TensorMap")
    p.add_argument("--output-factor", type=float, default=-1.0 / math.sqrt(3.0),
                   help="default: lambda=0 component -> isotropic trace/3")
    p.add_argument("--n-atoms", type=int, default=7)
    p.add_argument("--max-neighbors", type=int, default=11)
    p.add_argument("--graph-json", help="also write the GIR graph to this JSON file")
    a = p.parse_args()

    scripted, net = load_scripted(a.model)
    wrapper = NanoPETGraph(net, a.block, a.output_factor).eval()
    scale, comp = baselines(net, a.block_index, a.output_factor)
    species_map = {int(z): int(net.species_to_species_index[z]) for z in net.atomic_types}

    inputs, names, _ = example_export_args(a.n_atoms, a.max_neighbors, wrapper.cutoff, forces=True)
    n_dim = torch.export.Dim("n_atoms", min=1, max=max(2, a.n_atoms))
    m_dim = torch.export.Dim("max_neighbors", min=1, max=max(2, a.max_neighbors))
    e_dim = torch.export.Dim("n_edges", min=1, max=max(2, a.n_atoms * a.max_neighbors))
    nm = {0: n_dim, 1: m_dim}
    shapes = ({0: n_dim}, nm, nm, nm, {0: e_dim}, nm)
    graph, weights = export_torch_model(
        wrapper, inputs, output_path=None, input_names=names,
        input_dtypes={"species": "i32", "neighbor_species": "i32", "reverse_neighbor_index": "i32"},
        dynamic_shapes=shapes, strict=False)
    print(f"graph: {len(graph.nodes)} nodes, {len(weights)} weights; scale {scale}, "
          f"{len(comp)} composition entries, species {species_map}")

    if a.graph_json:
        Path(a.graph_json).write_text(json.dumps(graph.to_dict(), indent=1))
    w = GGUFWriter()
    w.add_string("general.architecture", "pet-graph")
    w.add_string("general.name", Path(a.model).stem)
    w.add_string("general.version", "nanopet")
    w.add_string("pet.target", a.block)
    w.add_float32("pet.output_factor", a.output_factor)
    w.add_float32("pet.cutoff", wrapper.cutoff)
    w.add_float32("pet.cutoff_width", wrapper.width)
    w.add_int32("pet.d_pet", wrapper.d)
    w.add_float32("pet.energy_scale", scale)
    w.add_string("pet.cutoff_function", "cosine")
    w.add_int32("pet.forces_mode", 1)
    w.add_int32("pet.nc_forces", 0)
    w.add_int32("pet.forward_only", 1)  # no gradients: predict without forces
    w.add_float32("pet.num_neighbors_adaptive", 0.0)
    w.add_array_int32("pet.species_map", [v for z in sorted(species_map) for v in (z, species_map[z])])
    if comp:
        w.add_array_int32("pet.composition_keys", list(comp))
        w.add_array_float32("pet.composition_values", list(comp.values()))
    w.add_string("graph.json", json.dumps(graph.to_dict()))
    w.add_string("graph.weight_shapes", json.dumps({k: list(t.shape) for k, t in weights.items()}))
    for k, t in weights.items():
        w.add_tensor(k, t, transpose_2d=False)
    w.write(a.output)
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
