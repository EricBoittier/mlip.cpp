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


def backbone_kind(name):
    """None (pet-jax's Backbone), 'nofinal' (NoFinalCombineBackbone) or 'layered' (LayeredBackbone), by class name."""
    if not name or name.endswith("petjax.model.Backbone"):
        return None
    for cls, kind in (("NoFinalCombineBackbone", "nofinal"), ("LayeredBackbone", "layered")):
        if name.split(".")[-1] == cls:
            return kind
    raise ValueError(f"custom backbone {name!r}: not supported (NoFinalCombineBackbone / LayeredBackbone only)")


class IrisTrunk(_AuxMixin, nn.Module):
    """The iris PET trunk as an mlip.cpp pet-graph (forces mode: distances and cutoff factors in-graph)."""

    def __init__(self, cfg):
        super().__init__()
        self._box = _Box()
        self.cfg = cfg
        d, dn, dh, Z = cfg["d_pet"], cfg["d_node"], cfg["d_head"], cfg["max_atomic_number"] + 1
        L = cfg["num_gnn_layers"]
        self.final_mp = cfg.get("final_message_passing", True)
        # his les-water backbones (ipsum-work play/les-water/backbones.py trunk(); iris explore/alessandro-strict-local
        # 1d412cf): NoFinalCombineBackbone = no combine after the last layer; LayeredBackbone = the standard trunk
        # returning per-layer features (entry l before layer l's combine, the last entry after the final combine). Both
        # build the comb_norms with flax's default LayerNorm eps 1e-6 (pet-jax's Backbone 1e-5)
        self.backbone = backbone_kind(cfg.get("backbone"))
        comb_eps = 1e-5
        if self.backbone == "nofinal":
            self.final_mp, comb_eps = False, 1e-6
        elif self.backbone == "layered":
            self.final_mp, comb_eps = True, 1e-6
        # iris PET (38c91c4): per_layer_readout = one Energy head per layers entry (energy_head_<l>), the energy their
        # sum, each optionally times a learned gate (layer_gate_<l>); feature_layer (PETLR charge_layer) = which
        # entry's node features the LR branch reads
        self.per_layer = bool(cfg.get("per_layer_readout"))
        self.gates = bool(cfg.get("learn_layer_gates"))
        self.feature_layer = cfg.get("charge_layer", cfg.get("feature_layer"))
        if (self.per_layer or self.feature_layer is not None) and self.backbone != "layered":
            raise ValueError("per_layer_readout / charge_layer need a backbone returning layers (LayeredBackbone)")
        self.n_heads = L + 1 if self.per_layer else 1
        self.cutoff, self.width = float(cfg["cutoff"]), float(cfg["cutoff_width"])
        self.neighbor_embedder = nn.Embedding(Z, d)  # pet-jax backbone/edge_embedder (initial messages)
        self.node_embedders = nn.ModuleList([nn.Embedding(Z, dn)])
        self.gnn_layers = nn.ModuleList([GNNLayer(cfg, i == 0) for i in range(L)])
        n_comb = L if self.final_mp else L - 1
        self.combination_norms = nn.ModuleList([nn.LayerNorm(2 * d, eps=comb_eps) for _ in range(n_comb)])
        self.combination_mlps = nn.ModuleList(
            [nn.Sequential(nn.Linear(2 * d, 2 * d), nn.SiLU(), nn.Linear(2 * d, d)) for _ in range(n_comb)])
        H = self.n_heads
        self.node_energy_heads = nn.ModuleList([_mlp2(dn, dh) for _ in range(H)])
        self.node_final_layers = nn.ModuleList([nn.Linear(dh, 1) for _ in range(H)])
        self.edge_energy_heads = nn.ModuleList([_mlp2(d, dh) for _ in range(H)])
        self.edge_final_layers = nn.ModuleList([nn.Linear(dh, 1) for _ in range(H)])
        if self.per_layer and self.gates:
            for l in range(H):
                setattr(self, f"layer_gate_{l}", nn.Parameter(torch.ones(1)))
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
        layers = []
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
            if self.backbone == "layered":  # his trunk(): the features before this layer's combine
                layers.append((node[:, 0, :], messages))
            if i < L - 1 or self.final_mp:
                rev = edge.reshape(n * k, d)[reverse_neighbor_index].reshape(n, k, d)
                rev = torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(rev), rev)
                comb = self.combination_mlps[i](self.combination_norms[i](torch.cat([edge, rev], dim=-1)))
                messages = messages + torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(comb), comb)
        node = node[:, 0, :]
        if self.backbone == "layered":
            layers.append((node, messages))  # the final entry, after the last combine

        def head(h, nd, msg):  # pet-jax Energy: node head + sum_j edge head x cutoff
            node_e = self.node_final_layers[h](self.node_energy_heads[h](nd))[:, 0]
            edge_e = self.edge_final_layers[h](self.edge_energy_heads[h](msg))[..., 0]
            edge_e = torch.where(padding_mask, torch.zeros_like(edge_e), edge_e)
            return node_e + (edge_e * cut).sum(dim=1)

        if self.per_layer:
            energy = 0.0
            for h, (nd, msg) in enumerate(layers):
                e_h = head(h, nd, msg)
                energy = energy + (e_h * getattr(self, f"layer_gate_{h}") if self.gates else e_h)
        else:
            energy = head(0, node, messages)
        feat = node if self.feature_layer is None else layers[self.feature_layer][0]
        self._aux = dict(node=feat, r=r, env=cut)  # for IrisPETLR (the trunk's own outputs, not graph outputs)
        return energy


# ---------------------------------------------------------------------------------------------------------------
# The long-range branch of iris PETLR with physical charges and charge_allocation="flows" (iris.pet.petlr
# `_PhysicalLRModuleTiled`, iris.pet.charges), open boundaries. Parameter names follow his flax tree `params/lr`.
# ---------------------------------------------------------------------------------------------------------------

EV_A = 14.399645478425667  # jaxpme.prefactors.eV_A


def softplus(z):
    """jax.nn.softplus = logaddexp(z, 0) = relu(z) + log1p(exp(-|z|)), with log1p by Goldberg's trick
    (log1p(u) = log(1 + u) u / ((1 + u) - 1), u where 1 + u rounds to 1): ggml has no log1p, and the plain
    log(1 + u) loses all relative accuracy of the tiny conductances at strongly negative z (in fp32: absolute 6e-8,
    times potentials of 1e5 in the seeded presets), where JAX's log1p keeps it."""
    u = torch.exp(-torch.abs(z))
    w = u + 1.0
    d = w - 1.0
    tiny = d <= 0.0
    l1p = torch.where(tiny, u, torch.log(w) * u / torch.where(tiny, torch.ones_like(d), d))
    return l1p + torch.relu(z)


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
        # conserving_charges (his feat/conserving-charges a98c2f6, iris.pet.charges there): the per-component Fermi
        # level ("fermi") or the residual allocation ("residual"), then ONE edge transfer, then the polarisation passes.
        # One component per structure only (his sample prep raises on more without fragment labels; mlip.cpp too), so
        # the component Fermi level is the structure's: the same maths as the flows fermi reference below
        self.conserving = bool(cfg.get("conserving_charges"))
        if self.conserving:
            if cfg.get("charge_allocation", "fermi") not in ("fermi", "residual"):
                raise ValueError("conserving_charges: charge_allocation fermi or residual")
            if not cfg.get("physical_charges") or cfg.get("num_charges", 1) != 1:
                raise ValueError("IrisLR conserving_charges: physical charges, num_charges=1 only")
            self.ref = "fermi" if cfg.get("charge_allocation", "fermi") == "fermi" else "residual"
            cfg = {**cfg, "transfer_passes": 0}
        elif cfg.get("charge_allocation") is None and cfg.get("physical_charges") and cfg.get("num_charges", 1) == 1:
            # the charge HEAD of his pre-flows physical branch (iris before 2d61e0c; the les-water ladder runs, iris
            # explore/alessandro-strict-local 1d412cf): q = MLP(h) * charge_scale (a learned scalar, or the constant
            # charge_scale_init), point charges, then the polarisation passes; no reference, no transfer
            # (charge_allocation None is his main branch's default too, a2658e3: the same head path there)
            for key in ("remove_sr_cutoff", "readout_phi0", "readout_charges", "polarisation_descent", "smearing",
                        "long_range_transfer", "induced_dipoles", "hellmann_feynman_forces"):
                if cfg.get(key):
                    raise ValueError(f"IrisLR head charges: {key} not ported")
            self.ref = "head"
            cfg = {**cfg, "transfer_passes": 0}
        else:
            self.ref = cfg.get("charge_reference", "species")
        if not self.conserving and self.ref != "head" and (
                cfg.get("charge_allocation") != "flows" or not cfg.get("physical_charges")
                or cfg.get("num_charges", 1) != 1):
            raise ValueError("IrisLR: physical charges, charge_allocation=flows and num_charges=1 only")
        # species_weighted / species_fill: his branch feat/long-range-transfer (3ed96b0, iris.pet.charges)
        if self.ref not in ("fermi", "species", "both", "species_weighted", "species_fill", "residual", "qeq",
                            "threshold", "head"):
            raise ValueError(f"IrisLR: charge_reference {self.ref!r} not yet ported")
        # long_range_transfer (feat/long-range-transfer): per transfer pass a drive a_i moved through the Coulomb kernel of
        # Gaussian charges (width smearing) over the whole structure, dq = G a - a (G 1)
        # a uniform external field (his feat/conserving-charges a98c2f6: the per-structure input `field` [3], V/A): the
        # pass drives' external potential difference -E . R_ij / 2 and the enthalpy -E . P_flow; single-component
        # conserving charges only here. The field force E . Z0 on the pass-1 charges is not part of the energy (his
        # iris.pet.predict adds it): see field_forces below; mlip.cpp refuses forces under a nonzero field
        self.field = "field" in (cfg.get("inputs") or [])
        if self.field and not self.conserving:
            raise ValueError("IrisLR: the field input is ported with conserving_charges only")
        # frozen Fermi level (his frozen_fermi_level, the Born charges' tapped charges): mu (fermi) or the residual
        # correction (residual) out of the gradient; never exported (the field-force oracle check only)
        self.frozen = False
        # induced dipoles: NOT ported. His path needs jax-pme EwaldDipole (lab-cosmo/jax-pme#38); its branch head 2a8d052
        # predates the pair layout his pinned jax-pme (1c26495) merged, and his a98c2f6 batching fails on it
        # (_flatten_jaxpme unpack): no oracle without rebasing that PR ourselves
        if cfg.get("induced_dipoles"):
            raise ValueError("IrisLR: induced_dipoles not ported (no oracle: jax-pme#38 is incompatible with his pin)")
        self.lrt = bool(cfg.get("long_range_transfer"))
        if self.lrt and cfg.get("smearing") is None:
            raise ValueError("long_range_transfer needs smearing")
        self.m_min, self.m_max, self.T = cfg.get("charge_state_min", -3), cfg.get("charge_state_max", 3), \
            cfg.get("charge_temperature", 0.1)
        self.cutoff, self.width, self.num_rbf = float(cfg["cutoff"]), float(cfg["cutoff_width"]), 8
        self.sigma = cfg.get("smearing")
        if self.ref in ("fermi", "both", "threshold"):
            self.score = nn.Linear(d, 1)
        if self.ref in ("species", "both", "species_weighted", "threshold"):
            self.species_charge = nn.Parameter(torch.zeros(cfg["max_atomic_number"] + 1))
        if self.ref == "head":  # masked MLP((d_hidden, d_hidden, 1)) (flax auto-name MLP_0), x charge_scale
            self.MLP_0 = _flax_mlp((d, dh, dh, 1))
            self.learn_scale = bool(cfg.get("learn_charge_scale", True))
            if self.learn_scale:
                self.charge_scale = nn.Parameter(torch.ones(()))
            else:
                self.charge_scale_const = float(cfg.get("charge_scale_init", 1.0))
        if self.ref == "threshold":  # his feat/threshold-reference (101575d): a learned threshold per species
            self.species_threshold = nn.Parameter(torch.zeros(cfg["max_atomic_number"] + 1))
        if self.ref in ("species_weighted", "threshold"):  # softplus per species, the share of the excess
            self.species_weight = nn.Parameter(torch.zeros(cfg["max_atomic_number"] + 1))
        if self.ref == "species_fill":  # one score and one softplus prefactor per species
            self.species_score = nn.Parameter(torch.zeros(cfg["max_atomic_number"] + 1))
            self.species_scale = nn.Parameter(torch.zeros(cfg["max_atomic_number"] + 1))
        if self.ref == "qeq":  # iris.pet.charges.qeq_reference (a2658e3): per-species hardness, electronegativity head
            if cfg.get("smearing") is None:
                raise ValueError("charge_reference='qeq' needs smearing")
            self.qeq_hardness = nn.Parameter(torch.zeros(cfg["max_atomic_number"] + 1))
            self.electronegativity = nn.Linear(d, 1)
            self.qeq_maxiter, self.qeq_tol = 20, 1e-5
        if self.ref == "residual":  # iris.pet.charges.residual_charges (conserving-charges branch)
            self.see_charge = bool(cfg.get("weights_see_charge"))
            self.free_charge = nn.Linear(d, 1)
            if self.see_charge:  # Q_c through a Gaussian basis, one centre per integer of the charge-state range
                self.allocation_hidden = nn.Linear(d + self.m_max - self.m_min + 1, dh)
                self.allocation_weight = nn.Linear(dh, 1)
            else:
                self.allocation_weight = nn.Linear(d, 1)
        if self.conserving:  # iris.pet.charges.transfers: one transfer, a_i, a_j, rbf (+ q0_i, q0_j)
            self.see_q0 = bool(cfg.get("transfers_see_charges"))
            self.transfer_embed = nn.Linear(d, dh)
            self.transfer_mlp = _flax_mlp((2 * dh + self.num_rbf + (2 if self.see_q0 else 0), dh, 1))
            self.transfer_gate = nn.Parameter(torch.zeros(()))
        self.weighted = self.ref == "species_fill"
        self.fermi = self.ref in ("fermi", "both", "species_fill")
        self.transfer_passes = int(cfg.get("transfer_passes", 1))
        for k in range(1, self.transfer_passes + 1):
            setattr(self, f"transfer_embed_{k}", nn.Linear(d, dh))
            setattr(self, f"transfer_mlp_{k}", _flax_mlp((2 * dh + self.num_rbf + 4, dh, 1)))
            setattr(self, f"transfer_gate_{k}", nn.Parameter(torch.zeros(())))
            if self.lrt:  # iris.pet.charges.long_range_drive
                setattr(self, f"long_range_embed_{k}", nn.Linear(d, dh))
                setattr(self, f"long_range_mlp_{k}", _flax_mlp((dh + 2, dh, 1)))
                setattr(self, f"long_range_gate_{k}", nn.Parameter(torch.zeros(())))
        self.passes = int(cfg.get("polarisation_passes", 1))
        self.shared = bool(cfg.get("polarisation_shared", True))
        self.form = cfg.get("conductance_form", "layerscale")
        self.cscale = float(cfg.get("conductance_scale", 1.0))
        self.cinit = float(cfg.get("conductance_init", -4.0))
        self.hard = bool(cfg.get("hardness", True))
        self.hardness_init = float(cfg.get("hardness_init", 2.0))  # (the head branch's parameter; 2.0 elsewhere)
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

    def extra_columns(self):
        """[(name, width)] of the graph's output columns after the first 8 (GGUF pet.iris_lr.columns): per-atom terms
        whose structure sum is the output: charge_residual (threshold reference)."""
        out = []
        if self.ref == "threshold":
            out.append(("charge_residual", 1))
        if self.conserving:
            out.append(("dipole_flow", 3))
        if self.field:
            # his field force E . Z0 (mlip.cpp, open structures): the pass-1 charges q0 and X, the quantity whose
            # gradient times a host weight is that of q0 at a frozen Fermi level (fermi: the scores s, weight f'(s - mu)
            # from column 7; residual: the free charges p, weight 1). Per atom, NOT summed by the host
            out += [("zf_q0", 1), ("zf_x", 1)]
        return out

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

    def scores(self, node, species=None):
        bound = 0.5 * (self.m_max - self.m_min) + 2.0
        if self.ref == "species_fill":  # (bound tanh(score / bound))[Z], as species_fill_reference
            return (bound * torch.tanh(self.species_score / bound))[species]
        return bound * torch.tanh(self.score(node)[:, 0] / bound)

    def weights(self, species):
        """species_fill: the prefactor c_Z = softplus(species_scale)[Z] (the Fermi level's weights); else None."""
        return softplus(self.species_scale)[species] if self.weighted else None

    def qeq(self, node, species, total_charge, potentials):
        """iris.pet.charges.qeq_reference (a2658e3) on one structure: q = Q / n + P z, z the jax.scipy cg solution
        (x0 = 0, maxiter 20, tol 1e-5 relative, atol 0, Jacobi preconditioner P (1 / (J + self)) P) of
        P A P z = -P (A Q / n + chi), A x = J x + K x, K x = 2 phi(x) + self x (Gaussian self-energy); the gradient as
        his jax.lax.custom_linear_solve: the cotangent of z solved by the same truncated cg (lambda), then
        lambda . (b - A z*) differentiated at z* fixed. Returns (q, the stop-gradient stationarity residual
        P (A q + chi)). Eager torch (a data-dependent iteration count): the fp64 reference, not traced."""
        self_energy = EV_A * math.sqrt(2.0 / math.pi) / self.sigma
        hard = softplus(self.qeq_hardness)[species]
        chi = self.electronegativity(node)[:, 0]
        n = chi.shape[0]
        project = lambda x: x - x.sum() / n
        apply = lambda x, J=hard: J * x + (2.0 * potentials(x) + self_energy * x)
        uniform = (total_charge / n) + torch.zeros_like(chi)
        b = -project(apply(uniform) + chi)
        with torch.no_grad():
            Jd = hard.detach()
            A0 = lambda x: project(apply(project(x), Jd))
            inv_d = 1.0 / (Jd + self_energy)
            M0 = lambda r: project(inv_d * r)
            z_val = jax_cg(A0, b.detach(), M0, self.qeq_maxiter, self.qeq_tol)
            self._qeq_fwd_iterations = jax_cg.last_iterations

        outer = self

        class _Solve(torch.autograd.Function):  # value 0; backward: the cg solve of the cotangent
            @staticmethod
            def forward(ctx, r):
                return torch.zeros_like(r)

            @staticmethod
            def backward(ctx, g):
                with torch.no_grad():
                    lam = jax_cg(A0, g, M0, self.qeq_maxiter, self.qeq_tol)
                    outer._qeq_bwd_iterations = jax_cg.last_iterations
                    return lam

        r = b - project(apply(project(z_val)))
        z = z_val + _Solve.apply(r)
        q = uniform + project(z)
        resid = project(apply(q) + chi).detach()
        self._qeq_iterations = (self._qeq_fwd_iterations, None)
        return q, resid

    def qeq_traced(self, node, species, total_charge, potentials, z, lam):
        """The graph form of `qeq` (mlip.cpp's three-stage evaluation): z [n] (the host cg solution, a gradient leaf)
        and lam [n] (the host cg solution of the cotangent of z, 0 in the first pass) are inputs. Returns (q, the
        stationarity residual P (A q + chi), lam * r per atom with r = b - P A P z): the loss sum E + sum lam r has
        the derivative of his custom_linear_solve at fixed (z, lam), and d(sum E)/dz with lam = 0 is the cotangent."""
        self_energy = EV_A * math.sqrt(2.0 / math.pi) / self.sigma
        hard = softplus(self.qeq_hardness)[species]
        chi = self.electronegativity(node)[:, 0]
        nt = torch.ones_like(chi).sum()  # n as a tensor (a symbolic size would be traced as an op on species)
        project = lambda x: x - x.sum() / nt
        apply = lambda x: hard * x + (2.0 * potentials(x) + self_energy * x)
        uniform = total_charge / nt + torch.zeros_like(chi)
        b = -project(apply(uniform) + chi)
        pz = project(z)
        r = b - project(apply(pz))
        q = uniform + pz
        resid = project(apply(q) + chi)
        return q, resid, lam * r

    def qeq_apply(self, species, potentials, x):
        """The host cg's operator pieces: (apply(x) = J x + 2 phi(x) + self x, the Jacobi diagonal J + self)."""
        self_energy = EV_A * math.sqrt(2.0 / math.pi) / self.sigma
        hard = softplus(self.qeq_hardness)[species]
        return hard * x + (2.0 * potentials(x) + self_energy * x), hard + self_energy

    def forward(self, node, species, positions, neighbor_index, padding_mask, reverse_neighbor_index, edge_vectors,
                env, total_charge, mu_bisect, gapped, m_round, top, bottom, dipole_origin, ewald=None, qeq_z=None,
                qeq_lambda=None, qeq_apply_x=None, field=None, comp=None):
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

        if ewald is None:
            # --- open-boundary potentials (jax-pme: prefactor * 1/2 sum_{j != i} q_j / r_ij) ---
            # one-sided broadcasts only (ggml binary ops repeat the second operand into the first)
            D = positions[None, :, :].expand(n, n, 3) - positions[:, None, :]
            D2 = (D * D).sum(dim=-1)
            R = torch.sqrt(D2 + 1e-15)
            # the self term out: D2 is exactly 0 on the diagonal only (bitwise the (1 - I) / (R + I) of before, and
            # traceable: no eye of a dynamic size)
            K = torch.where(D2 <= 0.0, torch.zeros_like(R), torch.ones_like(R) / R)  # no reciprocal op in the converter
            coulomb = lambda qq: 0.5 * EV_A * (K @ qq)
        else:
            coulomb = ewald_potentials(positions, ewald)

        def potentials(qq):
            phi = coulomb(qq)
            if self.sigma is not None:
                # erfc as 1 - erf (ggml has erf only; 1e-16 absolute in fp64, ~6e-8 in fp32 on a term ~1e-6 at 5 A)
                er = (-torch.erf(r / (math.sqrt(2.0) * self.sigma)) + 1.0) / torch.where(padding_mask, torch.ones_like(r), r) * keep
                phi = phi - 0.5 * EV_A * (er * gather_atoms(qq)).sum(dim=1)
            return phi

        if qeq_apply_x is not None:  # the host cg's operator graph (IrisQEqApplyGraph): no charges, no energy
            return self.qeq_apply(species, potentials, qeq_apply_x)
        # --- reference ---
        aux = {}
        if comp is not None:
            # MULTI-COMPONENT conserving charges (his per-component Fermi level / residual correction, iris.pet.charges
            # base_charges / residual_charges, a98c2f6), components and Q_c from the host (geometric components of the
            # neighbour graph and mlip.cpp's R-only Q_c rule, component_charges_rule): comp = (M [n, C] one-hot,
            # Q_c [C], mu_c [C] host bisection, gapped_c [C]); per-component sums seg(x) = sum_i M_ic x_i
            M, Qc, mu_c, gap_c = comp
            seg = lambda v: (M * v[:, None]).sum(dim=0)
            bc = lambda v: (M * v[None, :]).sum(dim=1)
            if self.ref == "fermi":
                s = self.scores(node, species)
                mub = bc(mu_c)
                mu_n = mu_c + (seg(self.occupation(s - mub)) - Qc) / torch.clamp(seg(self.density(s - mub)), min=1e-6)
                d_top, d_bot = seg((s - m_round) * top), seg((s - m_round) * bottom)
                mu = gap_c * (0.5 * (d_top + d_bot)) + (-gap_c + 1.0) * mu_n
                if self.frozen:
                    mu = mu.detach()
                mua = bc(mu)
                x = s - mua
                q = self.occupation(x)
                aux.update(scores=s, fermi_level=mua, metallicity=self.density(x).sum(),
                           density_atoms=self.density(x))
            else:  # residual
                p_free = self.free_charge(node)[:, 0]
                aux["free_atoms"] = p_free
                if self.see_charge:
                    Qa = bc(Qc)
                    basis = torch.cat([torch.exp(-2.0 * (Qa - float(m)) ** 2)[:, None]
                                       for m in range(self.m_min, self.m_max + 1)], dim=-1)  # [n, n_states]
                    xw = torch.cat([node, basis], dim=-1)
                    wq = softplus(self.allocation_weight(torch.nn.functional.silu(self.allocation_hidden(xw)))[:, 0])
                else:
                    wq = softplus(self.allocation_weight(node)[:, 0])
                deficit = Qc - seg(p_free)
                corr = wq / bc(torch.clamp(seg(wq), min=1e-6)) * bc(deficit)
                q = p_free + (corr.detach() if self.frozen else corr)
                atoms_c = torch.clamp(seg(torch.ones_like(p_free)), min=1.0)
                aux["residual_atoms"] = bc(torch.abs(deficit) / atoms_c)
        elif self.fermi:
            s = self.scores(node, species)
            w = self.weights(species)
            if w is None:
                mu_n = mu_bisect + ((self.occupation(s - mu_bisect)).sum() - total_charge) / torch.clamp(
                    self.density(s - mu_bisect).sum(), min=1e-6)
            else:  # fermi_level with weights: sum_i w_i f(s_i - mu) = Q, density sum_i w_i f'
                mu_n = mu_bisect + ((self.occupation(s - mu_bisect) * w).sum() - total_charge) / torch.clamp(
                    (self.density(s - mu_bisect) * w).sum(), min=1e-6)
            # top / bottom: indicator weights of the extremes of s - m, divided by the tie count, as JAX's max / min
            # derivative splits over ties
            d_top, d_bot = ((s - m_round) * top).sum(), ((s - m_round) * bottom).sum()
            mu = gapped * (0.5 * (d_top + d_bot)) + (-gapped + 1.0) * mu_n
            if self.frozen:
                mu = mu.detach()
            x = s - mu
            q = self.occupation(x)
            if w is None:
                aux.update(scores=s, fermi_level=mu, metallicity=self.density(x).sum())
            else:  # q0 = c f(x), metallicity sum_i c_i f'(x_i)
                q = w * q
                dens_atoms = w * self.density(x)
                aux.update(scores=s, fermi_level=mu, metallicity=dens_atoms.sum(), density_atoms=dens_atoms)
            if self.ref == "both":
                c = self.species_charge[species]
                q = q + c - c.sum() / torch.ones_like(c).sum()  # / n as a tensor (a symbolic size traced as a species DIV)
        elif self.ref == "qeq" and qeq_z is not None:  # the graph form (mlip.cpp): z, lam from the host
            q, resid, adj = self.qeq_traced(node, species, total_charge, potentials, qeq_z, qeq_lambda)
            aux["qeq_adjoint"] = adj
            aux["qeq_residual"] = torch.sqrt((resid * resid).sum())
            aux["residual_atoms"] = resid * resid / torch.clamp(aux["qeq_residual"], min=1e-30)
        elif self.ref == "qeq":
            q, resid = self.qeq(node, species, total_charge, potentials)
            # his qeq_residual [S] = sqrt(sum_i resid_i^2) (eV), as a per-atom column that sums to it
            aux["qeq_residual"] = torch.sqrt((resid * resid).sum())
            aux["residual_atoms"] = resid * resid / torch.clamp(aux["qeq_residual"], min=1e-30)
        elif self.ref == "residual":  # q0 = p + w (Q - sum p) / max(sum w, 1e-6); residual |Q - sum p|
            p_free = self.free_charge(node)[:, 0]
            aux["free_atoms"] = p_free
            if self.see_charge:
                basis = torch.cat([torch.exp(-2.0 * (total_charge.reshape(1) - float(m)) ** 2)
                                   for m in range(self.m_min, self.m_max + 1)])  # [n_states], one structure
                xw = torch.cat([node, basis[None, :].expand(n, basis.shape[0])], dim=-1)
                wq = softplus(self.allocation_weight(torch.nn.functional.silu(self.allocation_hidden(xw)))[:, 0])
            else:
                wq = softplus(self.allocation_weight(node)[:, 0])
            deficit = total_charge - p_free.sum()
            corr = wq / torch.clamp(wq.sum(), min=1e-6) * deficit
            q = p_free + (corr.detach() if self.frozen else corr)
            # his per-atom residual |deficit| / atoms_c, summed per structure (one component): |deficit|; as a per-atom
            # column (graph output 7) that sums to it
            aux["residual_atoms"] = torch.abs(deficit) / torch.ones_like(p_free).sum() + torch.zeros_like(p_free)
        elif self.ref == "head":
            scale = self.charge_scale if self.learn_scale else self.charge_scale_const
            q = self.MLP_0(node)[:, 0] * scale
        elif self.ref == "threshold":
            # iris.pet.charges.threshold_reference (101575d): q = c_Z + f(s_i - theta_Z), spread by the softplus
            # species weights to sum to Q: q0 = q - w (sum q - Q) / max(sum w, 1e-12); the residual R = sum q - Q
            # (charge_residual [S]) and metallicity sum_i f'(s_i - theta_Z). No Fermi level, no host bisection
            s = self.scores(node, species)
            x = s - self.species_threshold[species]
            qt = self.species_charge[species] + self.occupation(x)
            wz = softplus(self.species_weight)[species]
            resid = qt.sum() - total_charge
            q = qt - wz * (resid / torch.clamp(wz.sum(), min=1e-12))
            dens_atoms = self.density(x)
            aux.update(scores=s, metallicity=dens_atoms.sum(), density_atoms=dens_atoms, charge_residual=resid,
                       # per-atom terms that sum to R (graph output column charge_residual)
                       charge_residual_atoms=qt - total_charge / torch.ones_like(qt).sum())
        elif self.ref == "species_weighted":  # q0 = c_Z - w_Z (sum_j c_Z_j - Q) / sum_j w_Z_j
            c = self.species_charge[species]
            wz = softplus(self.species_weight)[species]
            q = c - wz * ((c.sum() - total_charge) / torch.clamp(wz.sum(), min=1e-12))
        else:
            c = self.species_charge[species]
            q = c - (c.sum() - total_charge) / torch.ones_like(c).sum()  # / n as a tensor, as above
        aux["charges_base"] = q
        if self.conserving:  # the one transfer of iris.pet.charges.transfers (conserving-charges branch)
            aux["charges_pass1"] = q
            aux["zf_q0_atoms"] = q
            aux["zf_x_atoms"] = aux["scores"] if self.ref == "fermi" else aux["free_atoms"]
            a = self.transfer_embed(node)
            parts = [a[:, None, :].expand(n, k, a.shape[1]), gather_atoms(a), rbf]
            if self.see_q0:
                parts += [q[:, None, None].expand(n, k, 1), gather_atoms(q)[..., None]]
            t = self.transfer_mlp(torch.cat(parts, dim=-1))[..., 0] * self.transfer_gate
            q = q + flow_sum(t)
            drives = t  # his `drives`: the transfer's edge values, then each pass's drive (dipole_flow, enthalpy)
        # --- transfer passes: edge MLP on node embeddings, charges and SR potentials ---
        if self.sigma is None:
            kern = keep / torch.where(padding_mask, torch.ones_like(r), r)
        else:
            kern = torch.erf(r / (math.sqrt(2.0) * self.sigma)) / torch.where(padding_mask, torch.ones_like(r), r) * keep
        kernel_sr = weight * kern
        if self.lrt:
            # the long-range transfer kernel G (his flow_potential: jax-pme RawPotential, prefactor 1, potentials x 2):
            # open: erf(r / (sqrt 2 sigma)) / r over all pairs i != j; periodic: the reciprocal-space sum alone (k = 0
            # out, no self / background / slab terms), at the batch's Ewald smearing (the calculator's lr_k2 argument)
            if ewald is None:
                Dl = positions[None, :, :].expand(n, n, 3) - positions[:, None, :]
                D2l = (Dl * Dl).sum(dim=-1)
                Rl = torch.sqrt(D2l + 1e-15)
                Gl = torch.where(D2l <= 0.0, torch.zeros_like(Rl), torch.erf(Rl / (math.sqrt(2.0) * self.sigma)) / Rl)
                conv = lambda x: Gl @ x
            else:
                conv = ewald_reciprocal(positions, ewald)
            ones_g = conv(torch.ones_like(q))
        moved = torch.zeros_like(q)
        for p in range(1, self.transfer_passes + 1):
            phi_sr = (kernel_sr * gather_atoms(q)).sum(dim=1)
            a = getattr(self, f"transfer_embed_{p}")(node)
            xin = torch.cat([a[:, None, :].expand(n, k, a.shape[1]), gather_atoms(a), rbf,
                             q[:, None, None].expand(n, k, 1), gather_atoms(q)[..., None],
                             phi_sr[:, None, None].expand(n, k, 1), gather_atoms(phi_sr)[..., None]], dim=-1)
            t = getattr(self, f"transfer_mlp_{p}")(xin)[..., 0] * getattr(self, f"transfer_gate_{p}")
            step = flow_sum(t)
            if self.lrt:
                h = getattr(self, f"long_range_embed_{p}")(node)
                xl = torch.cat([h, q[:, None], phi_sr[:, None]], dim=-1)
                al = getattr(self, f"long_range_mlp_{p}")(xl)[:, 0] * getattr(self, f"long_range_gate_{p}")
                step = step + (conv(al) - al * ones_g)
            q = q + step
            moved = moved + step
        aux["transfers"] = moved
        phi = potentials(q)
        extra = torch.zeros_like(q)
        if field is not None:
            proj = (edge_vectors * field[None, None, :]).sum(dim=-1)  # E . R_ij per slot
        if self.passes > 1:
            if not self.conserving:
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
                    sigma_c = softplus(z)
                if field is not None:  # potentials are half the physical potential: the field enters as -E . R / 2
                    drive = sigma_c * (gather_atoms(phi) - phi[:, None] - 0.5 * proj)
                else:
                    drive = sigma_c * (gather_atoms(phi) - phi[:, None])
                if self.conserving:
                    drives = drives + drive
                step = flow_sum(drive)
                if self.hard:
                    zh = getattr(self, f"hardness_head{sx}")(node)[:, 0] + self.hardness_init
                    eta = softplus(zh)
                    extra = extra + 0.5 * eta * step * step
                delta = delta + step
                q = q + step
                phi = potentials(q)
            aux["delta_charges"] = delta
        if self.conserving:
            # dipole_flow = -1/2 sum_e F_e R_e over the antisymmetrised flows F = (d_ij - d_ji) / 2 w_ij of the summed
            # drives (the transfer and every pass): a per-atom column; under a field the enthalpy
            # 1/2 sum_j F_ij E . R_ij per atom (= -E . P_flow summed)
            d_rev = drives.reshape(n * k)[rev_flat].reshape(n, k)
            flow = 0.5 * (drives - d_rev) * weight
            aux["dipole_flow_atoms"] = -0.5 * (flow[..., None] * edge_vectors).sum(dim=1)
            if field is not None:
                extra = extra + 0.5 * (flow * proj).sum(dim=1)
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


def jax_cg(A, b, M, maxiter, tol, atol=0.0):
    """jax.scipy.sparse.linalg.cg's _cg_solve (x0 = 0): the same recurrences and stopping rule (|r|^2 > max(tol^2
    |b|^2, atol^2) and k < maxiter, |r|^2 from r itself with a preconditioner)."""
    x = torch.zeros_like(b)
    atol2 = max(tol * tol * float((b * b).sum()), atol * atol)
    r = b - A(x)
    z = M(r)
    p, gamma, k = z, (r * z).sum(), 0
    while float((r * r).sum()) > atol2 and k < maxiter:
        Ap = A(p)
        alpha = gamma / (p * Ap).sum()
        x = x + alpha * p
        r = r - alpha * Ap
        z = M(r)
        gamma_ = (r * z).sum()
        p = z + (gamma_ / gamma) * p
        gamma, k = gamma_, k + 1
    jax_cg.last_iterations = k
    return x


def ewald_potentials(positions, e):
    """jax-pme batched_tiled `Ewald(prefactor=eV_A).potentials` (1c26495) as iris calls it (`iris.pet.long_range`):
    half the physical potential of point charges in a 3-D periodic cell or a 2-D slab, at his calculator's frozen
    parameters. Returns q -> phi [n]. `e` holds (all tensors; the derivative inputs of the virial marked *):
      lr_index [n, kl] long   atom j of each real-space slot (FULL list at rc + skin: each pair both ways, so
                              0.5 sum over slots = his half list summed both ways), padded slots any index
      lr_shift [n, kl, 3] *   the slot's lattice shift S @ cell (A)
      lr_keep  [n, kl]        1 real slot, 0 padded
      kvec     [K, 3]   *     k-vectors (halfspace grid n @ 2 pi cell^-T, k = 0 excluded), his `generate_ewald_kvectors`
      volume   [1]      *     |det cell| (the effective cell for a slab)
      smearing [1], cutoff [1] (rc; slots at r >= rc are masked, as his `_masked`)
      slab [1] (1 for a slab), slab_normal [3], slab_basis [1] (|cell[k] . n|, the effective cell's)
    Real space: sr(r) = 1/r - erf(r / (s sqrt 2)) / r; k space: sum_k W_k Re(exp(-i k r_i) S_k), W = 2 G / V with
    G = 4 pi exp(-s^2 k^2 / 2) / k^2 (halfspace g = 2; the k = 0 term is 0 for Coulomb); corrections: self
    -q sqrt(2 / pi) / s, background -2 pi s^2 Q / V, slab (Yeh-Berkowitz) 4 pi (z M - (M2 + Q z^2) / 2 - Q b^2 / 12) / V;
    (k space + corrections) / 2; everything times the prefactor."""
    n, kl = e["lr_index"].shape
    pos_j = positions[e["lr_index"].reshape(n * kl)].reshape(n, kl, 3)
    D = pos_j - positions[:, None, :] + e["lr_shift"]
    D2 = (D * D).sum(dim=-1)
    pad = e["lr_keep"] <= 0.0
    r = torch.sqrt(torch.where(pad, torch.ones_like(D2), D2))
    keep = torch.where(r - e["cutoff"] >= 0.0, torch.zeros_like(r), e["lr_keep"])  # (ge with a tensor: unmapped)
    one_s = torch.ones_like(e["smearing"]) / e["smearing"]  # 1 / s (no reciprocal op in the converter)
    inv_s = one_s * (1.0 / math.sqrt(2.0))
    v_sr = (-torch.erf(r * inv_s) + 1.0) / r * keep  # 1/r - erf/r as (1 - erf) / r (ggml has erf only)
    kvec, V, s = e["kvec"], e["volume"], e["smearing"]
    k2 = (kvec * kvec).sum(dim=-1)
    W = 2.0 * (4.0 * math.pi) * torch.exp(-0.5 * (s * s) * k2) / k2 / V
    theta = positions @ kvec.transpose(0, 1)  # [n, K]
    c, sn = torch.cos(theta), torch.sin(theta)
    zn = positions @ e["slab_normal"]  # [n]

    def phi(qq):
        q_j = qq[e["lr_index"].reshape(n * kl)].reshape(n, kl)
        rs = 0.5 * (v_sr * q_j).sum(dim=1)
        Sr, Si = c.transpose(0, 1) @ qq, sn.transpose(0, 1) @ qq  # [K] (matrix @ vector)
        ks = c @ (W * Sr) + sn @ (W * Si)
        Q = qq.sum()
        corr = -qq * (math.sqrt(2.0 / math.pi) * one_s) - 2.0 * (math.pi * s * s) * Q / V
        M, M2 = (qq * zn).sum(), (qq * zn * zn).sum()
        b = e["slab_basis"]
        slab = 4.0 * math.pi * (zn * M - 0.5 * (M2 + Q * zn * zn) - Q / 12.0 * b * b) / V
        return EV_A * (rs + 0.5 * (ks + corr + e["slab"] * slab))
    return phi


def ewald_reciprocal(positions, e):
    """The k-space sum of `ewald_potentials` alone, prefactor 1 and the 1/2 removed: x -> sum_k W_k Re(exp(-i k r_i)
    S_k(x)) [n] (his long-range transfer kernel on a periodic structure, iris.pet.charges.flow_potential)."""
    kvec, V, s = e["kvec"], e["volume"], e["smearing"]
    k2 = (kvec * kvec).sum(dim=-1)
    W = 2.0 * (4.0 * math.pi) * torch.exp(-0.5 * (s * s) * k2) / k2 / V
    theta = positions @ kvec.transpose(0, 1)
    c, sn = torch.cos(theta), torch.sin(theta)
    return lambda x: c @ (W * (c.transpose(0, 1) @ x)) + sn @ (W * (sn.transpose(0, 1) @ x))


def bisect_fermi(lr, s, total, iterations=40, eps=1e-6, w=None):
    """Host side of iris `fermi_level` on frozen scores s [n] (numpy / torch double): (mu_bisect, gapped, m_round,
    top, bottom). The in-graph part (Newton step, gapped midpoint) is IrisLR.forward. `w` [n]: the weights of
    species_fill (sum_i w_i f(s_i - mu) = Q; integers within eps)."""
    s = torch.as_tensor(s, dtype=torch.float64)
    wt = torch.ones_like(s) if w is None else torch.as_tensor(w, dtype=torch.float64)
    margin = 1.0 + 40.0 * lr.T
    lo, hi = float(s.min()) - lr.m_max - margin, float(s.max()) - lr.m_min + margin
    lo, hi = torch.tensor(lo, dtype=s.dtype), torch.tensor(hi, dtype=s.dtype)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        above = ((lr.occupation(s - mid) * wt).sum() - total) > 0
        lo, hi = (mid, hi) if above else (lo, mid)
    mu = 0.5 * (lo + hi)
    m = torch.round(lr.occupation(s - mu))
    d = s - m
    top = (d == d.max()).to(s.dtype); top = top / top.sum()
    bottom = (d == d.min()).to(s.dtype); bottom = bottom / bottom.sum()
    mid = 0.5 * float(d.max() + d.min())
    gapped = (abs(float((m * wt).sum()) - total) < (0.5 if w is None else eps)) and float(d.max() - d.min()) < 1.0 \
        and float((lr.density(s - mid) * wt).sum()) < eps
    return float(mu), gapped, m, top, bottom


class IrisPETLRGraph(IrisTrunk):
    """Trunk + LR branch as ONE mlip.cpp graph (after M. Langer, ipsum/PET-LR). Inputs: the trunk's (species,
    neighbor_species, edge_vectors, padding_mask, reverse_neighbor_index, cutoff_values[, charge_index, spin_index]),
    then neighbor_index [n, k] (atom of each slot), positions [n, 3], total_charge [1], the host side of fermi_level
    (mu_bisect [1], gapped [1], m_round [n], top [n], bottom [n]) and dipole_origin [3].
    Output [n, 8]: per-atom energy (energy_scale applied in-graph to the trunk, not to the LR energy, as his PETLR),
    charge, the dipole contribution q_i (r_i - r_0) [3], mu (every row), score s_i, occupation density f'(s_i - mu).
    The energy column alone is differentiated for forces / virial."""

    N_OUT = 8  # without extra columns; n_out = N_OUT + the widths of lr.extra_columns()

    def __init__(self, trunk, lr, energy_scale):
        # an IrisTrunk itself (trunk parameter names as in the trunk-only GGUF, within GGUF's 64-char limit)
        super().__init__(trunk.cfg)
        self.load_state_dict(trunk.state_dict())
        self.lr = lr
        self.energy_scale = nn.Parameter(torch.tensor([float(energy_scale)]))
        self.n_cond = 2 if trunk.conditioning else 0
        self.n_out = self.N_OUT + sum(w for _, w in lr.extra_columns())

    def forward(self, species, neighbor_species, edge_vectors, *args):
        return self._forward(species, neighbor_species, edge_vectors, args, None)

    def _forward(self, species, neighbor_species, edge_vectors, args, ew):
        trunk_args = args[:3 + self.n_cond]
        lr_args = args[3 + self.n_cond:]
        (neighbor_index, positions, total_charge, mu_bisect, gapped, m_round, top, bottom,
         dipole_origin) = lr_args[:9]
        # QEq: the host cg's solution z and the cotangent solution lam (QEQ_INPUTS, after dipole_origin)
        qz, qlam = lr_args[9:11] if self.lr.ref == "qeq" else (None, None)
        field = lr_args[9] if self.lr.field else None  # [3] (FIELD_INPUTS, after dipole_origin)
        # multi-component (IrisPETLRGraphMulti): comp_onehot [n, C], comp_charge [C], comp_mu [C], comp_gapped [C] last
        comp = tuple(lr_args[-4:]) if getattr(self, "multi", False) else None
        e_sr = IrisTrunk.forward(self, species, neighbor_species, edge_vectors, *trunk_args) * self.energy_scale
        node, env = self._aux["node"], self._aux["env"]
        padding_mask, reverse = args[0], args[1]
        e_lr = self.lr(node, species, positions, neighbor_index, padding_mask, reverse, edge_vectors, env,
                       total_charge, mu_bisect, gapped, m_round, top, bottom, dipole_origin, ewald=ew, qeq_z=qz,
                       qeq_lambda=qlam, field=field, comp=comp)
        aux = self.lr._aux
        q = aux["charges"]
        zero = torch.zeros_like(q)
        mu = aux.get("fermi_level")
        mu_col = (zero + mu) if mu is not None else zero  # (multi: each atom's component's mu)
        # QEq: column 6 is lam_i r_i, the adjoint term mlip.cpp adds to the loss (not an output)
        s_col = aux["scores"] if "scores" in aux else aux["qeq_adjoint"] if "qeq_adjoint" in aux else zero
        dens = aux["residual_atoms"] if "residual_atoms" in aux else aux["density_atoms"] if "density_atoms" in aux else (
            self.lr.density(s_col - mu) if mu is not None else zero)
        dip = (positions - dipole_origin[None, :]) * q[:, None]
        cols = [(e_sr + e_lr)[:, None], q[:, None], dip, mu_col[:, None], s_col[:, None], dens[:, None]]
        # extra columns (pet.iris_lr.columns, after the 8 above): per-atom terms the host sums over the structure
        for name, width in self.lr.extra_columns():
            v = aux[name + "_atoms"]
            cols.append(v[:, None] if width == 1 else v)
        return torch.cat(cols, dim=-1)


class IrisPETLRGraphPBC(IrisPETLRGraph):
    """IrisPETLRGraph for periodic cells (3-D or slabs): the open all-pair Coulomb replaced by his jax-pme Ewald
    (`ewald_potentials`). Extra inputs after dipole_origin: lr_index [n, n_lr] (i32), lr_shift [n, n_lr, 3],
    lr_keep [n, n_lr], kvec [n_kvec, 3], volume [1], smearing [1], cutoff [1], slab [1], slab_normal [3],
    slab_basis [1]. Same parameters (and names) as IrisPETLRGraph: one GGUF carries both graphs."""

    EWALD = ("lr_index", "lr_shift", "lr_keep", "kvec", "volume", "smearing", "cutoff", "slab", "slab_normal",
             "slab_basis")

    def forward(self, species, neighbor_species, edge_vectors, *args):
        ew = dict(zip(self.EWALD, args[-len(self.EWALD):]))
        return self._forward(species, neighbor_species, edge_vectors, args[:-len(self.EWALD)], ew)


class IrisPETLRGraphMulti(IrisPETLRGraph):
    """IrisPETLRGraph for conserving charges on structures of SEVERAL components (the connected components of the
    mutual neighbour graph): extra inputs after the others, MULTI = comp_onehot [n, C] (f32 one-hot), comp_charge [C]
    (the host's Q_c), comp_mu [C] (the host's per-component bisection), comp_gapped [C]; mu_bisect / gapped unused.
    A separate graph, so the single-component one stays bit for bit as before."""

    MULTI = ("comp_onehot", "comp_charge", "comp_mu", "comp_gapped")
    multi = True


class IrisPETLRGraphMultiPBC(IrisPETLRGraphMulti):
    """IrisPETLRGraphMulti on periodic cells (the Ewald inputs after MULTI)."""

    def forward(self, species, neighbor_species, edge_vectors, *args):
        E = IrisPETLRGraphPBC.EWALD
        ew = dict(zip(E, args[-len(E):]))
        return self._forward(species, neighbor_species, edge_vectors, args[:-len(E)], ew)


def component_charges_rule(lr, comp, n_comp, Q, s=None, p=None):
    """mlip.cpp's R-only Q_c (no fragment labels; his code needs them on more than one component): integers summing to
    Q by the LARGEST-REMAINDER rounding of a real per-component share qhat_c: fermi, qhat_c = sum_{i in c} f(s_i - mu)
    at the structure's single Fermi level mu (bisect_fermi on the frozen scores, as the one-component reference);
    residual, qhat_c = sum_{i in c} p_i + (Q - sum p) n_c / n (the free charges, the deficit by atom count).
    floor(qhat) then +1 to the largest fractional parts (ties: the lower component index) until the sum is Q.
    numpy in double; the host (graph_model.cpp) is the same."""
    import numpy as np
    comp = np.asarray(comp)
    if s is not None:
        mu, _, _, _, _ = bisect_fermi(lr, torch.as_tensor(s, dtype=torch.float64), float(Q))
        f = lr.occupation(torch.as_tensor(s, dtype=torch.float64) - float(mu)).numpy()
        qhat = np.bincount(comp, weights=f, minlength=n_comp)
    else:
        p = np.asarray(p, float)
        qhat = np.bincount(comp, weights=p, minlength=n_comp) + (Q - p.sum()) * np.bincount(comp, minlength=n_comp) / len(p)
    base = np.floor(qhat)
    rest = int(round(Q - base.sum()))
    frac = qhat - base
    order = sorted(range(n_comp), key=lambda c: (-frac[c], c))
    for c in order[:max(rest, 0)]:
        base[c] += 1
    return base


class IrisScoresGraph(IrisTrunk):
    """The first pass of the two-pass mu: trunk + score head (forward only), output s [n] (fermi / both references)."""

    def __init__(self, trunk, lr):
        super().__init__(trunk.cfg)
        self.load_state_dict(trunk.state_dict())
        self.lr = lr

    def forward(self, species, neighbor_species, edge_vectors, *args):
        IrisTrunk.forward(self, species, neighbor_species, edge_vectors, *args)
        if self.lr.ref == "qeq":  # QEq: the electronegativity chi [n] (the host cg's right-hand side)
            return self.lr.electronegativity(self._aux["node"])[:, 0]
        if self.lr.ref == "residual":  # conserving residual: the free charges p [n] (the multi-component Q_c rule)
            return self.lr.free_charge(self._aux["node"])[:, 0]
        s = self.lr.scores(self._aux["node"], species)
        w = self.lr.weights(species)
        # species_fill: [n, 2] = (s, c) (the host bisection's weights); else s [n]
        return s if w is None else torch.cat([s[:, None], w[:, None]], dim=-1)


class IrisQEqApplyGraph(nn.Module):
    """The operator of the host cg of QEq (mlip.cpp's three-stage evaluation, iris.pet.charges.qeq_reference): no
    trunk. Inputs species, edge_vectors, padding_mask, reverse_neighbor_index, neighbor_index, positions, qeq_x [n]
    (+ the EWALD inputs of IrisPETLRGraphPBC when `periodic`); output [n, 2] = (apply(x) = J x + 2 phi(x) + self x,
    the Jacobi diagonal J + self). The host projects (P x = x - mean x) and runs the cg in double."""

    INPUTS = ("species", "edge_vectors", "padding_mask", "reverse_neighbor_index", "neighbor_index", "positions",
              "qeq_x")

    def __init__(self, lr, periodic=False):
        super().__init__()
        self.lr = lr
        self.periodic = periodic

    def forward(self, species, edge_vectors, padding_mask, reverse, neighbor_index, positions, x, *ewald):
        ew = dict(zip(IrisPETLRGraphPBC.EWALD, ewald)) if self.periodic else None
        env = torch.zeros_like(edge_vectors[..., 0])
        z1 = torch.zeros_like(x[:1])
        a, d = self.lr(None, species, positions, neighbor_index, padding_mask, reverse, edge_vectors, env, z1, z1, z1,
                       x, x, x, torch.zeros_like(positions[0]), ewald=ew, qeq_apply_x=x)
        return torch.cat([a[:, None], d[:, None]], dim=-1)
