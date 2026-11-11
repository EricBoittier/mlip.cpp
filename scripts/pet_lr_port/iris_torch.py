"""Pure-torch transliteration of the iris / pet-jax PET trunk (pet-jax c3f216c `petjax.model.Backbone` + `Energy`,
as mounted by `iris.pet.model.PET`), after M. Langer, ipsum/PET-LR. No metatrain: this module is only the tracing
vehicle for mlip.cpp's FX exporter (`export_pytorch.fx_converter`), built directly from the iris parameter tree.

Graph contract (that of mlip.cpp's pet-graph GGUFs, so the C++ side needs no change): inputs
  species [n], neighbor_species [n, k], edge_vectors [n, k, 3], padding_mask [n, k] (True = padded),
  reverse_neighbor_index [n*k], cutoff_values [n, k] (per-pair cutoff radius), and, with conditioning,
  charge_index [n] (table row of the charge, c + max_charge) and spin_index [n] (2S+1 - 1);
output: per-atom energies [n] (energy_scale is applied by the C++ side from the GGUF metadata).
Parameter names follow mlip.cpp's exporter wrapper (`node_embedders.0`, `neighbor_embedder`, `gnn_layers.L.*`,
`combination_{norms,mlps}.L`, `node_energy_heads.0`, `node_final_layers.0`, `edge_*`, `system_conditioning.*`),
so the GGUF tensor names and the C++ graph hooks (node features = input of node_energy_heads_0_0) are unchanged.

Semantics carried from pet-jax that metatrain's PET cannot express:
  - `final_message_passing=False`: no reverse-edge combine after the last GNN layer (the edge messages that
    reach the edge head are then `messages + edge`);
  - any `num_gnn_layers` (incl. 1), plain cutoff (hard: cutoff_values = cutoff) or adaptive per-pair cutoffs;
  - species tables indexed by Z (row Z, `max_atomic_number + 1` rows), as iris stores them.
Attention follows pet-jax: logits + log(clip(cutoff, 1e-15)); padded tokens are excluded through a cutoff of 0
(log 1e-15 = -34.5: the excluded weight is exp(-34.5) relative, i.e. 1e-15, below fp64 resolution of the sums).
"""
import math

import torch
from torch import nn


class SwiGLU(nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.w_in = nn.Linear(d_in, 2 * d_hidden)
        self.w_out = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        v, g = self.w_in(x).chunk(2, dim=-1)
        return self.w_out(v * torch.sigmoid(g))


class RMSNorm(nn.Module):
    """torch.nn.RMSNorm with eps = finfo(dtype).eps (pet-jax `rms_norm`); traced as RMS_NORM by fx_converter."""

    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return torch.nn.functional.rms_norm(x, (x.shape[-1],), self.weight, torch.finfo(x.dtype).eps)


class Attention(nn.Module):
    def __init__(self, d, heads, temperature):
        super().__init__()
        self.input_linear = nn.Linear(d, 3 * d)
        self.output_linear = nn.Linear(d, d)
        self.heads, self.temperature = heads, temperature

    def forward(self, x, log_cut):
        n, t, f = x.shape
        h = self.heads
        x = self.input_linear(x).reshape(n, t, 3, h, f // h).permute(2, 0, 3, 1, 4)
        q, k, v = x[0], x[1], x[2]
        w = torch.matmul(q, k.transpose(-2, -1)) / ((f // h) ** 0.5 * self.temperature) + log_cut
        w = w.softmax(dim=-1)
        return self.output_linear(torch.matmul(w, v).transpose(1, 2).reshape(n, t, f))


class TransLayer(nn.Module):
    """pet-jax TransformerLayer (PreLN, RMSNorm, SwiGLU); d_node != d_pet expands the centre token."""

    def __init__(self, d_pet, d_node, d_ff, heads, temperature):
        super().__init__()
        self.attention = Attention(d_pet, heads, temperature)
        self.norm_attention = RMSNorm(d_pet)
        self.norm_mlp = RMSNorm(d_pet)
        self.mlp = SwiGLU(d_pet, d_ff, d_pet)
        self.expanded = d_node != d_pet
        if self.expanded:
            self.center_contraction = nn.Linear(d_node, d_pet)
            self.center_expansion = nn.Linear(d_pet, d_node)
            self.norm_center_features = RMSNorm(d_node)
            self.center_mlp = SwiGLU(d_node, 2 * d_node, d_node)

    def forward(self, node, edge, log_cut):
        proj = self.center_contraction(node) if self.expanded else node
        tokens = torch.cat([proj, edge], dim=1)
        out = self.attention(self.norm_attention(tokens), log_cut)
        out_node, out_edge = torch.split(out, [1, out.shape[1] - 1], dim=1)
        if self.expanded:
            out_node = node + self.center_expansion(out_node)
            out_node = out_node + self.center_mlp(self.norm_center_features(out_node))
        else:
            out_node = node + out_node
        out_edge = edge + out_edge
        out_edge = out_edge + self.mlp(self.norm_mlp(out_edge))
        return out_node, out_edge


class Trans(nn.Module):
    def __init__(self, n, *a):
        super().__init__()
        self.layers = nn.ModuleList([TransLayer(*a) for _ in range(n)])


class GNNLayer(nn.Module):
    def __init__(self, cfg, first):
        super().__init__()
        d = cfg["d_pet"]
        self.first = first
        self.edge_embedder = nn.Linear(4, d)  # pet-jax gnn_layers_L_edge_embed: [R_ij, r_ij] -> d_pet
        self.compress = nn.Sequential(nn.Linear((2 if first else 3) * d, d), nn.SiLU(), nn.Linear(d, d))
        if not first:
            self.neighbor_embedder = nn.Embedding(cfg["max_atomic_number"] + 1, d)
        self.trans = Trans(cfg["num_attention_layers"], d, cfg["d_node"], cfg["d_feedforward"],
                           cfg["num_heads"], cfg["attention_temperature"])


def _mlp2(d_in, d_hidden):
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.SiLU(), nn.Linear(d_hidden, d_hidden), nn.SiLU())


class IrisTrunk(nn.Module):
    """The iris PET trunk as an mlip.cpp pet-graph (forces mode: distances and cutoff factors in-graph)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, dn, dh, Z = cfg["d_pet"], cfg["d_node"], cfg["d_head"], cfg["max_atomic_number"] + 1
        L = cfg["num_gnn_layers"]
        self.final_mp = cfg.get("final_message_passing", True)
        self.cutoff, self.width = float(cfg["cutoff"]), float(cfg["cutoff_width"])
        self.neighbor_embedder = nn.Embedding(Z, d)  # pet-jax backbone/edge_embedder (initial messages)
        self.node_embedders = nn.ModuleList([nn.Embedding(Z, dn)])
        self.gnn_layers = nn.ModuleList([GNNLayer(cfg, i == 0) for i in range(L)])
        n_comb = L if self.final_mp else L - 1
        self.combination_norms = nn.ModuleList([nn.LayerNorm(2 * d, eps=1e-5) for _ in range(n_comb)])
        self.combination_mlps = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d)) for _ in range(n_comb)])
        self.node_energy_heads = nn.ModuleList([_mlp2(dn, dh)])
        self.node_final_layers = nn.ModuleList([nn.Linear(dh, 1)])
        self.edge_energy_heads = nn.ModuleList([_mlp2(d, dh)])
        self.edge_final_layers = nn.ModuleList([nn.Linear(dh, 1)])
        self.conditioning = bool(cfg.get("system_conditioning", False))
        if self.conditioning:
            self.max_charge = int(cfg["max_charge"])
            self.max_spin_multiplicity = int(cfg["max_spin_multiplicity"])
            sc = nn.Module()
            sc.charge_embedding = nn.Embedding(2 * self.max_charge + 1, dn)
            sc.spin_multiplicity_embedding = nn.Embedding(self.max_spin_multiplicity, dn)
            sc.project = nn.Sequential(nn.Linear(2 * dn, dn), nn.SiLU(), nn.Linear(dn, dn))
            self.system_conditioning = sc

    def cutoff_factors(self, r, cut):
        """pet-jax cutoff_bump(r, cut, width), written without tan (no ggml equivalent)."""
        x = torch.clamp((r - (cut - self.width)) / self.width, 0.0, 1.0)
        # pi as a Python float: torch.tensor(math.pi) is a float32 constant (8.7e-8 relative), which alone made
        # an fp64 trace differ from pet-jax by 3e-9
        a = math.pi * torch.clamp(x, min=1e-6, max=1.0 - 1e-6)
        return 0.5 * (1.0 + torch.tanh(torch.cos(a) / torch.sin(a)))

    def forward(self, species, neighbor_species, edge_vectors, *args):
        # args = (padding_mask, reverse_neighbor_index, cutoff_values[, charge_index, spin_index]): the
        # exporter's dynamic-shapes layout
        padding_mask, reverse_neighbor_index, cutoff_values = args[:3]
        cond_idx = args[3:]
        n, k = neighbor_species.shape
        d = self.cfg["d_pet"]
        keep = ~padding_mask
        zero = torch.zeros_like(edge_vectors[..., :1])
        r = torch.sqrt((edge_vectors * edge_vectors).sum(dim=-1))
        cut = self.cutoff_factors(r, cutoff_values)
        cut = torch.where(padding_mask, torch.zeros_like(cut), cut)
        # token cutoffs [n, 1, 1 + k]: the centre token 1, padded tokens 0; log-bias as pet-jax
        tok = torch.cat([torch.ones_like(cut[:, :1]), cut], dim=1)
        log_cut = torch.log(torch.clamp(tok, min=1e-15))
        # pet-jax masks padded tokens out of the softmax entirely (flax mask), not by the log(1e-15) bias alone
        # (metatrain): their weight is then exactly 0 instead of 1e-15 exp(logit), which is visible beyond fp32
        tok_pad = torch.cat([torch.zeros_like(padding_mask[:, :1]), padding_mask], dim=1)
        log_cut = torch.where(tok_pad, torch.full_like(log_cut, -1e30), log_cut)[:, None, None, :]
        cond = None
        if self.conditioning:
            sc = self.system_conditioning
            cond = sc.project(torch.cat([sc.charge_embedding(cond_idx[0]),
                                         sc.spin_multiplicity_embedding(cond_idx[1])], dim=-1))
        messages = self.neighbor_embedder(neighbor_species.flatten()).view(n, k, d)
        messages = torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(messages), messages)
        node = self.node_embedders[0](species)[:, None, :]
        geom = torch.cat([edge_vectors, r[..., None]], dim=-1)
        L = len(self.gnn_layers)
        for i, layer in enumerate(self.gnn_layers):
            g = layer.edge_embedder(geom)
            g = torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(g), g)
            if layer.first:
                x = torch.cat([g, messages], dim=-1)
            else:
                nb = layer.neighbor_embedder(neighbor_species)
                nb = torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(nb), nb)
                x = torch.cat([g, nb, messages], dim=-1)
            edge = layer.compress(x)
            edge = torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(edge), edge)
            for tl in layer.trans.layers:
                node, edge = tl(node, edge, log_cut)
            if cond is not None:
                node = node + cond[:, None, :]
            edge = torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(edge), edge)
            messages = messages + edge
            if i < L - 1 or self.final_mp:
                rev = edge.reshape(n * k, d)[reverse_neighbor_index].reshape(n, k, d)
                rev = torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(rev), rev)
                comb = self.combination_mlps[i](self.combination_norms[i](torch.cat([edge, rev], dim=-1)))
                messages = messages + torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(comb), comb)
        node = node[:, 0, :]
        node_e = self.node_final_layers[0](self.node_energy_heads[0](node))[:, 0]
        edge_e = self.edge_final_layers[0](self.edge_energy_heads[0](messages))[..., 0]
        edge_e = torch.where(padding_mask, torch.zeros_like(edge_e), edge_e)
        return node_e + (edge_e * cut).sum(dim=1)
