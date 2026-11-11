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


class _Box:
    """Opaque holder of forward-time scratch (torch.export snapshots module attributes as a pytree: a dict
    attribute that gains keys during forward fails its check, an opaque object does not)."""


class _AuxMixin:
    @property
    def _aux(self):
        return self.__dict__.setdefault("_box", _Box()).__dict__.get("aux")

    @_aux.setter
    def _aux(self, v):
        self.__dict__.setdefault("_box", _Box()).aux = v


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


class IrisTrunk(_AuxMixin, nn.Module):
    """The iris PET trunk as an mlip.cpp pet-graph (forces mode: distances and cutoff factors in-graph)."""

    def __init__(self, cfg):
        super().__init__()
        self._box = _Box()
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
        # zeros_like - 1e30, not full_like: the FX converter has no full_like (it passed log_cut through, so the
        # traced graph used the -34.5 bias; invisible in fp32, 1e-15 relative)
        log_cut = torch.where(tok_pad, torch.zeros_like(log_cut) - 1e30, log_cut)[:, None, None, :]
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
        energy = node_e + (edge_e * cut).sum(dim=1)
        self._aux = dict(node=node, r=r, env=cut)  # for IrisPETLR (the trunk's own outputs, not graph outputs)
        return energy


# ---------------------------------------------------------------------------------------------------------------
# The long-range branch of iris PETLR with physical charges and charge_allocation="flows" (iris.pet.petlr
# `_PhysicalLRModuleTiled`, iris.pet.charges), open boundaries. Parameter names follow his flax tree `params/lr`.
# ---------------------------------------------------------------------------------------------------------------

EV_A = 14.399645478425667  # jaxpme.prefactors.eV_A


def _flax_mlp(sizes):
    """flax petjax.MLP: Dense -> SiLU -> ... -> Dense (no activation on the last)."""
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.SiLU()]
    return nn.Sequential(*layers[:-1])


class IrisLR(_AuxMixin, nn.Module):
    """`_PhysicalLRModuleTiled` for the configurations of his flows runs: num_charges = 1, charge_reference fermi /
    species / both, transfer passes with the edge-MLP drive, polarisation passes (shared or not, layerscale or softplus
    conductance, hardness), readout_charges / readout_phi0, point or Gaussian (smearing) charges, open boundaries."""

    def __init__(self, cfg):
        super().__init__()
        self._box = _Box()
        self.cfg = cfg
        d = cfg["d_node"]
        dh = d if cfg.get("d_lr") is None else cfg["d_lr"]
        dm = cfg["d_feedforward"] if cfg.get("d_lr") is None else cfg["d_lr"]
        self.ref = cfg.get("charge_reference", "species")
        if cfg.get("charge_allocation") != "flows" or not cfg.get("physical_charges") or cfg.get("num_charges", 1) != 1:
            raise ValueError("IrisLR: physical charges, charge_allocation=flows and num_charges=1 only")
        if self.ref not in ("fermi", "species", "both"):
            raise ValueError(f"IrisLR: charge_reference {self.ref!r} not yet ported")
        self.m_min, self.m_max, self.T = cfg.get("charge_state_min", -3), cfg.get("charge_state_max", 3), \
            cfg.get("charge_temperature", 0.1)
        self.cutoff, self.width, self.num_rbf = float(cfg["cutoff"]), float(cfg["cutoff_width"]), 8
        self.sigma = cfg.get("smearing")
        if self.ref in ("fermi", "both"):
            self.score = nn.Linear(d, 1)
        if self.ref in ("species", "both"):
            self.species_charge = nn.Parameter(torch.zeros(cfg["max_atomic_number"] + 1))
        self.transfer_passes = int(cfg.get("transfer_passes", 1))
        for k in range(1, self.transfer_passes + 1):
            setattr(self, f"transfer_embed_{k}", nn.Linear(d, dh))
            setattr(self, f"transfer_mlp_{k}", _flax_mlp((2 * dh + self.num_rbf + 4, dh, 1)))
            setattr(self, f"transfer_gate_{k}", nn.Parameter(torch.zeros(())))
        self.passes = int(cfg.get("polarisation_passes", 1))
        self.shared = bool(cfg.get("polarisation_shared", True))
        self.form = cfg.get("conductance_form", "layerscale")
        self.cscale = float(cfg.get("conductance_scale", 1.0))
        self.cinit = float(cfg.get("conductance_init", -4.0))
        self.hard = bool(cfg.get("hardness", True))
        for k in range(1, (self.passes if not self.shared else min(self.passes, 2))):
            sx = "" if k == 1 else f"_{k}"
            setattr(self, f"pol_embed{sx}", nn.Linear(d, dh))
            if self.form == "layerscale":
                setattr(self, f"pol_gate_in{sx}", nn.Linear(2 * dh + self.num_rbf, dh))
                setattr(self, f"pol_gate{sx}", nn.Linear(dh, 1))
            else:
                setattr(self, f"pol_conductance{sx}", _flax_mlp((2 * dh + self.num_rbf, dh, 1)))
            if self.hard:
                setattr(self, f"hardness_head{sx}", nn.Linear(d, 1))
        self.readout = bool(cfg.get("readout_charges")) or bool(cfg.get("readout_phi0"))
        if cfg.get("readout_phi0"):
            raise ValueError("IrisLR: readout_phi0 not yet ported")
        if self.readout:
            n_in = 1
            self.MLP_0 = _flax_mlp((n_in, dm, d))
            self.LayerNorm_0 = nn.LayerNorm(d, eps=1e-6)
            self.MLP_1 = _flax_mlp((d, dm, d))
            self.LayerNorm_1 = nn.LayerNorm(d, eps=1e-6)
            self.MLP_2 = _flax_mlp((d, dh, dh, 1))
            self.lr_scale = nn.Parameter(torch.zeros(1))

    # -- charge states (iris.pet.charges) --
    def steps(self):
        return [m - 0.5 for m in range(self.m_min + 1, self.m_max + 1)]

    def occupation(self, x):
        return self.m_min + sum(torch.sigmoid((x - c) / self.T) for c in self.steps())

    def density(self, x):
        out = 0
        for c in self.steps():
            s = torch.sigmoid((x - c) / self.T)
            out = out + s * (-s + 1.0)  # not 1.0 - s: rsub is not in the FX converter (bitwise the same)
        return out / self.T

    def scores(self, node):
        bound = 0.5 * (self.m_max - self.m_min) + 2.0
        return bound * torch.tanh(self.score(node)[:, 0] / bound)

    def forward(self, node, species, positions, neighbor_index, padding_mask, reverse_neighbor_index, edge_vectors,
                env, total_charge, mu_bisect, gapped, m_round, top, bottom, dipole_origin):
        """Per-atom LR energy [n] and the charges. Inputs beyond the trunk's: neighbor_index [n, k] (atom index of
        each slot), positions [n, 3], total_charge [1], the host's frozen-score bisection mu_bisect [1] and gapped
        decision (gapped [1] in {0, 1}, m_round [n] rounded occupations, top / bottom [n] indicator weights of the
        extremes of s - m divided by their tie counts), dipole_origin [3]. `env` is the trunk's cutoff_bump at the pair cutoffs (padded slots 0)."""
        n, k = neighbor_index.shape
        zero_e = torch.zeros_like(edge_vectors[..., 0])
        keep = torch.where(padding_mask, zero_e, zero_e + 1.0)  # (~mask).to(dtype), in mapped ops
        rev_flat = reverse_neighbor_index
        r = torch.sqrt((edge_vectors * edge_vectors).sum(dim=-1) + 1e-15)  # pet-jax safe_norm
        # _edge_geometry: symmetric taper, zero where the reverse slot is padded
        keep_rev = keep.reshape(n * k)[rev_flat].reshape(n, k)
        env_rev = env.reshape(n * k)[rev_flat].reshape(n, k)
        weight = 0.5 * (env + env_rev) * keep * keep_rev
        mus = [self.cutoff * i / (self.num_rbf - 1) for i in range(self.num_rbf)]
        # cat of [..., 1] slices, not torch.stack (the FX exporter has no stack)
        rbf = torch.cat([torch.exp(-(((r - m) * self.num_rbf / self.cutoff) ** 2))[..., None] for m in mus], dim=-1)

        def gather_atoms(x):  # x [n, ...] at each slot's neighbour -> [n, k, ...]
            return x[neighbor_index.reshape(n * k)].reshape((n, k) + tuple(x.shape[1:]))

        def flow_sum(t):  # sum_j 0.5 (t_ij - t_ji) w_ij
            t_rev = t.reshape(n * k)[rev_flat].reshape(n, k)
            return (0.5 * (t - t_rev) * weight).sum(dim=1)

        # --- reference ---
        aux = {}
        if self.ref in ("fermi", "both"):
            s = self.scores(node)
            mu_n = mu_bisect + ((self.occupation(s - mu_bisect)).sum() - total_charge) / torch.clamp(
                self.density(s - mu_bisect).sum(), min=1e-6)
            # top / bottom: indicator weights of the extremes of s - m, divided by the tie count, as JAX's max / min
            # derivative splits over ties
            d_top, d_bot = ((s - m_round) * top).sum(), ((s - m_round) * bottom).sum()
            mu = gapped * (0.5 * (d_top + d_bot)) + (-gapped + 1.0) * mu_n
            x = s - mu
            q = self.occupation(x)
            aux.update(scores=s, fermi_level=mu, metallicity=self.density(x).sum())
            if self.ref == "both":
                c = self.species_charge[species]
                q = q + c - c.sum() / torch.ones_like(c).sum()  # / n as a tensor (a symbolic size traced as a species DIV)
        else:
            c = self.species_charge[species]
            q = c - (c.sum() - total_charge) / torch.ones_like(c).sum()  # / n as a tensor, as above
        aux["charges_base"] = q
        # --- transfer passes: edge MLP on node embeddings, charges and SR potentials ---
        if self.sigma is None:
            kern = keep / torch.where(padding_mask, torch.ones_like(r), r)
        else:
            kern = torch.erf(r / (math.sqrt(2.0) * self.sigma)) / torch.where(padding_mask, torch.ones_like(r), r) * keep
        kernel_sr = weight * kern
        moved = torch.zeros_like(q)
        for p in range(1, self.transfer_passes + 1):
            phi_sr = (kernel_sr * gather_atoms(q)).sum(dim=1)
            a = getattr(self, f"transfer_embed_{p}")(node)
            xin = torch.cat([a[:, None, :].expand(n, k, a.shape[1]), gather_atoms(a), rbf,
                             q[:, None, None].expand(n, k, 1), gather_atoms(q)[..., None],
                             phi_sr[:, None, None].expand(n, k, 1), gather_atoms(phi_sr)[..., None]], dim=-1)
            t = getattr(self, f"transfer_mlp_{p}")(xin)[..., 0] * getattr(self, f"transfer_gate_{p}")
            step = flow_sum(t)
            q = q + step
            moved = moved + step
        aux["transfers"] = moved
        # --- open-boundary potentials (jax-pme: prefactor * 1/2 sum_{j != i} q_j / r_ij) ---
        # one-sided broadcasts only (ggml binary ops repeat the second operand into the first)
        D = positions[None, :, :].expand(n, n, 3) - positions[:, None, :]
        D2 = (D * D).sum(dim=-1)
        R = torch.sqrt(D2 + 1e-15)
        # the self term out: D2 is exactly 0 on the diagonal only (bitwise the (1 - I) / (R + I) of before, and
        # traceable: no eye of a dynamic size)
        K = torch.where(D2 <= 0.0, torch.zeros_like(R), torch.ones_like(R) / R)  # no reciprocal op in the converter

        def potentials(qq):
            phi = 0.5 * EV_A * (K @ qq)
            if self.sigma is not None:
                # erfc as 1 - erf (ggml has erf only; 1e-16 absolute in fp64, ~6e-8 in fp32 on a term ~1e-6 at 5 A)
                er = (-torch.erf(r / (math.sqrt(2.0) * self.sigma)) + 1.0) / torch.where(padding_mask, torch.ones_like(r), r) * keep
                phi = phi - 0.5 * EV_A * (er * gather_atoms(qq)).sum(dim=1)
            return phi

        phi = potentials(q)
        extra = torch.zeros_like(q)
        if self.passes > 1:
            aux["charges_pass1"] = q
            delta = torch.zeros_like(q)
            for p in range(1, self.passes):
                kk = 1 if self.shared else p
                sx = "" if kk == 1 else f"_{kk}"
                b = getattr(self, f"pol_embed{sx}")(node)
                bi, bj = b[:, None, :].expand(n, k, b.shape[1]), gather_atoms(b)
                sym = torch.cat([bi + bj, bi * bj, rbf], dim=-1)
                if self.form == "layerscale":
                    hidden = torch.nn.functional.silu(getattr(self, f"pol_gate_in{sx}")(sym))
                    sigma_c = self.cscale * getattr(self, f"pol_gate{sx}")(hidden)[..., 0]
                else:
                    z = getattr(self, f"pol_conductance{sx}")(sym)[..., 0] + self.cinit
                    sigma_c = torch.log(1.0 + torch.exp(-torch.abs(z))) + torch.relu(z)  # softplus
                drive = sigma_c * (gather_atoms(phi) - phi[:, None])
                step = flow_sum(drive)
                if self.hard:
                    zh = getattr(self, f"hardness_head{sx}")(node)[:, 0] + 2.0
                    eta = torch.log(1.0 + torch.exp(-torch.abs(zh))) + torch.relu(zh)
                    extra = extra + 0.5 * eta * step * step
                delta = delta + step
                q = q + step
                phi = potentials(q)
            aux["delta_charges"] = delta
        aux["charges"] = q
        aux["dipole"] = ((positions - dipole_origin[None, :]) * q[:, None]).sum(dim=0)
        energy = q * phi + extra
        if self.readout:
            h = node + self.MLP_0(q[:, None])
            h = self.LayerNorm_0(h)
            h = h + self.MLP_1(h)
            h = self.LayerNorm_1(h)
            energy = energy + self.MLP_2(h)[:, 0] * self.lr_scale
        self._aux = aux
        return energy


def bisect_fermi(lr, s, total, iterations=40, eps=1e-6):
    """Host side of iris `fermi_level` on frozen scores s [n] (numpy / torch double): (mu_bisect, gapped, m_round,
    top, bottom). The in-graph part (Newton step, gapped midpoint) is IrisLR.forward."""
    s = torch.as_tensor(s, dtype=torch.float64)
    margin = 1.0 + 40.0 * lr.T
    lo, hi = float(s.min()) - lr.m_max - margin, float(s.max()) - lr.m_min + margin
    lo, hi = torch.tensor(lo, dtype=s.dtype), torch.tensor(hi, dtype=s.dtype)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        above = (lr.occupation(s - mid).sum() - total) > 0
        lo, hi = (mid, hi) if above else (lo, mid)
    mu = 0.5 * (lo + hi)
    m = torch.round(lr.occupation(s - mu))
    d = s - m
    top = (d == d.max()).to(s.dtype); top = top / top.sum()
    bottom = (d == d.min()).to(s.dtype); bottom = bottom / bottom.sum()
    mid = 0.5 * float(d.max() + d.min())
    gapped = (abs(float(m.sum()) - total) < 0.5) and float(d.max() - d.min()) < 1.0 and \
        float(lr.density(s - mid).sum()) < eps
    return float(mu), gapped, m, top, bottom


class IrisPETLRGraph(IrisTrunk):
    """Trunk + LR branch as ONE mlip.cpp graph (after M. Langer, ipsum/PET-LR). Inputs: the trunk's (species,
    neighbor_species, edge_vectors, padding_mask, reverse_neighbor_index, cutoff_values[, charge_index, spin_index]),
    then neighbor_index [n, k] (atom of each slot), positions [n, 3], total_charge [1], the host side of fermi_level
    (mu_bisect [1], gapped [1], m_round [n], top [n], bottom [n]) and dipole_origin [3].
    Output [n, 8]: per-atom energy (energy_scale applied in-graph to the trunk, not to the LR energy, as his PETLR),
    charge, the dipole contribution q_i (r_i - r_0) [3], mu (every row), score s_i, occupation density f'(s_i - mu).
    The energy column alone is differentiated for forces / virial."""

    N_OUT = 8

    def __init__(self, trunk, lr, energy_scale):
        # an IrisTrunk itself (trunk parameter names as in the trunk-only GGUF, within GGUF's 64-char limit)
        super().__init__(trunk.cfg)
        self.load_state_dict(trunk.state_dict())
        self.lr = lr
        self.energy_scale = nn.Parameter(torch.tensor([float(energy_scale)]))
        self.n_cond = 2 if trunk.conditioning else 0

    def forward(self, species, neighbor_species, edge_vectors, *args):
        trunk_args = args[:3 + self.n_cond]
        (neighbor_index, positions, total_charge, mu_bisect, gapped, m_round, top, bottom,
         dipole_origin) = args[3 + self.n_cond:]
        e_sr = IrisTrunk.forward(self, species, neighbor_species, edge_vectors, *trunk_args) * self.energy_scale
        node, env = self._aux["node"], self._aux["env"]
        padding_mask, reverse = args[0], args[1]
        e_lr = self.lr(node, species, positions, neighbor_index, padding_mask, reverse, edge_vectors, env,
                       total_charge, mu_bisect, gapped, m_round, top, bottom, dipole_origin)
        aux = self.lr._aux
        q = aux["charges"]
        zero = torch.zeros_like(q)
        mu = aux.get("fermi_level")
        mu_col = (zero + mu) if mu is not None else zero
        s_col = aux["scores"] if "scores" in aux else zero
        dens = self.lr.density(s_col - mu) if mu is not None else zero
        dip = (positions - dipole_origin[None, :]) * q[:, None]
        return torch.cat([(e_sr + e_lr)[:, None], q[:, None], dip, mu_col[:, None], s_col[:, None], dens[:, None]],
                         dim=-1)


class IrisScoresGraph(IrisTrunk):
    """The first pass of the two-pass mu: trunk + score head (forward only), output s [n] (fermi / both references)."""

    def __init__(self, trunk, lr):
        super().__init__(trunk.cfg)
        self.load_state_dict(trunk.state_dict())
        self.lr = lr

    def forward(self, species, neighbor_species, edge_vectors, *args):
        IrisTrunk.forward(self, species, neighbor_species, edge_vectors, *args)
        return self.lr.scores(self._aux["node"])
