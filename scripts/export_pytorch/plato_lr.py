"""Export support for PET with Plato-PET-LR components (metatrain ``plato_lr`` hypers,
``metatrain/pet/modules/plato_lr.py``).

Four custom ops (namespace ``mlipcpp``) stand for the parts the graph interpreter builds
natively (``src/runtime/pet_plato_lr.cpp``); everything else (gates, charge heads, the
mesh LayerNorm and its down / up projections) stays ordinary traced PyTorch:

- ``rope_frame_attn(q, k, v, bias, pos, omega, frames, scale)``: the frame-averaged RoPE
  attention ``sum_f w_f softmax(RoPE_f q . RoPE_f k scale + bias) v`` of
  :class:`FrameRoPEAttention` (uniform ``w_f``); PET's own attention ``o_0`` and the gate
  ``o = o_0 + g_h (. - o_0)`` are traced around it, so ``o_0`` is the fused PET attention
  of a plain export.
- ``edge_group_sum(x, W, frames)``: ``mean_f SiLU(W R_f^T x[..., :3])`` of
  :class:`GroupSumEdgeEmbedder`.
- ``plato_mesh(v, filt..., k_quad, w_quad, ...)``: the mesh (periodic: B-spline spread, rfftn,
  learned filter, irfftn, gather, times the number of mesh points; open: the real-space
  kernel sum) on the latent channels ``v`` of :class:`PETMeshCoupling`.
- ``plato_les(q, charge_unit, ...)``: the per-atom LES energies of the latent charges
  (neutralised per structure, or shifted to the structure's total charge).

The positions, cell and total charge are not graph inputs: the C++ side supplies them to
the native builders (the energy depends on them through those ops only), and the
reference implementations below read them from :data:`CONTEXT` (eager checks).

The mesh FrameLinears have one frame, so each is folded to one ``Linear``
(``W_eff = sum_b phi_b(I) W_b``) on the export wrapper (``mesh_down`` / ``mesh_up``); a
GGUF written by ``mlipcpp-train`` is mapped back by :func:`unfold_mesh` (``W_b += phi_b
(W_eff' - W_eff) / |phi|^2``, the least-norm change of the basis weights).
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn.functional as Fn

# per-call data of the eager reference implementations (one structure): positions
# (n, 3), cell (3, 3), periodic (bool), charge (float, total charge in e)
CONTEXT: Dict[str, object] = {}
# modules the reference implementations call into, by key
_REGISTRY: Dict[str, torch.nn.Module] = {}


def _apply_rope(x, ang):
    x1, x2 = x.chunk(2, -1)
    c, s = ang.cos(), ang.sin()
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1)


@torch.library.custom_op("mlipcpp::rope_frame_attn", mutates_args=())
def rope_frame_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor,
                    pos: torch.Tensor, omega: torch.Tensor, frames: torch.Tensor,
                    scale: float) -> torch.Tensor:
    """q, k, v (N, H, L, Ch), bias (N, 1, L, L), pos (N, L, 3), omega (Ch/2, 3), frames
    (F, 3, 3): (N, H, L, Ch)."""
    R = frames.to(pos.dtype)
    Rw = torch.einsum("fab,cb->fca", R, omega.to(pos.dtype))
    ang = torch.einsum("nlx,fcx->nflc", pos, Rw)[:, :, None]
    qf = _apply_rope(q[:, None], ang)
    kf = _apply_rope(k[:, None], ang)
    a = qf @ kf.transpose(-1, -2) * scale + bias[:, None]
    return (torch.softmax(a, -1).mean(1) @ v).contiguous()


@rope_frame_attn.register_fake
def _(q, k, v, bias, pos, omega, frames, scale):
    return torch.empty_like(q)


@torch.library.custom_op("mlipcpp::edge_group_sum", mutates_args=())
def edge_group_sum(x: torch.Tensor, W: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
    """x (..., 4) = [r, |r|], W (d, 3), frames (F, 3, 3): mean_f SiLU(W R_f^T r), (..., d)."""
    r_f = torch.einsum("fji,...j->...fi", frames.to(x.dtype), x[..., :3])
    return Fn.silu(Fn.linear(r_f, W)).mean(-2)


@edge_group_sum.register_fake
def _(x, W, frames):
    return x.new_empty(list(x.shape[:-1]) + [W.shape[0]])


@torch.library.custom_op("mlipcpp::plato_mesh", mutates_args=())
def plato_mesh(v: torch.Tensor, filt0_w: torch.Tensor, filt0_b: torch.Tensor, filt2_w: torch.Tensor,
               filt2_b: torch.Tensor, k_quad: torch.Tensor, w_quad: torch.Tensor, sigma: float,
               spacing: float, open_boundary: int, key: str) -> torch.Tensor:
    """Latent channels v (n, C) -> the mesh potential (n, C) of PETMeshCoupling (physical
    normalisation); open_boundary 1: real-space sum, 0: zero for open structures."""
    mod = _REGISTRY.get(key)
    if mod is None or "positions" not in CONTEXT:
        return torch.zeros_like(v)
    pos = CONTEXT["positions"].to(v.dtype)
    if CONTEXT.get("periodic", False):
        cell = CONTEXT["cell"].to(v.dtype)
        M = [max(8, int(math.ceil(float(a) / spacing))) for a in torch.linalg.norm(cell, dim=-1)]
        return mod.coupling.mesh_batched(v[None], pos[None], cell[None], None, M, 1)[0] * float(M[0] * M[1] * M[2])
    if not open_boundary:
        return torch.zeros_like(v)
    return mod.realspace(v, pos)


@plato_mesh.register_fake
def _(v, filt0_w, filt0_b, filt2_w, filt2_b, k_quad, w_quad, sigma, spacing, open_boundary, key):
    return torch.empty_like(v)


@torch.library.custom_op("mlipcpp::plato_les", mutates_args=())
def plato_les(q: torch.Tensor, charge_unit: torch.Tensor, sigma: float, kcut: float,
              total_charge_constraint: int) -> torch.Tensor:
    """Latent charges q (n, Q) -> per-atom LES energies (n,), model units."""
    from metatrain.experimental.plato_pet_lr.modules.ewald import les_energy_finite, les_energy_periodic
    if "positions" not in CONTEXT:
        return q.new_zeros(q.shape[0])
    pos = CONTEXT["positions"].to(q.dtype)
    shift = q.mean(0)
    if total_charge_constraint:
        Q = float(CONTEXT.get("charge", 0.0)) / charge_unit.to(q.dtype)
        shift = shift - Q / q.shape[0]
    qs = q - shift
    if CONTEXT.get("periodic", False):
        return les_energy_periodic(qs, pos, CONTEXT["cell"].to(q.dtype), sigma, kcut)
    return les_energy_finite(qs, pos, sigma)


@plato_les.register_fake
def _(q, charge_unit, sigma, kcut, total_charge_constraint):
    return q.new_empty(q.shape[0])


# ---------------------------------------------------------------- module patches


def _rope_attention_forward(self, x, cutoff_factors, use_manual_attention, token_positions=None):
    """FrameRoPEAttention.forward for the export: PET's attention exactly as
    AttentionBlock (so the interpreter fuses it), the frame path as one custom op."""
    from metatrain.pet.modules.transformer import manual_attention
    initial_shape = x.shape
    x = self.input_linear(x)
    x = x.reshape(initial_shape[0], initial_shape[1], 3, self.num_heads, self.head_dim)
    x = x.permute(2, 0, 3, 1, 4)
    queries, keys, values = x[0], x[1], x[2]
    attn_weights = torch.clamp(cutoff_factors[:, None, :, :], self.epsilon)
    attn_weights = torch.log(attn_weights)
    if not use_manual_attention:
        raise NotImplementedError("plato_lr export needs the forces (manual attention) graph")
    o = manual_attention(queries, keys, values, attn_weights, self.temperature)
    if token_positions is not None:
        scale = 1.0 / (self.head_dim ** 0.5 * self.temperature)
        fa = torch.ops.mlipcpp.rope_frame_attn(queries, keys, values, attn_weights, token_positions,
                                               self.rope_omega, self.rope_frames, scale)
        o = o + self.rope_gate[None, :, None, None] * (fa - o)
    x = o.transpose(1, 2).reshape(initial_shape)
    return self.output_linear(x)


def _group_sum_forward(self, x):
    y = Fn.linear(x, self.weight, self.bias)
    return y + self.edge_gate * torch.ops.mlipcpp.edge_group_sum(x, self.vec.weight, self.group_frames)


def patch_modules() -> None:
    """Route FrameRoPEAttention and GroupSumEdgeEmbedder through the custom ops (export
    process only)."""
    from metatrain.pet.modules.plato_lr import FrameRoPEAttention, GroupSumEdgeEmbedder
    FrameRoPEAttention.forward = _rope_attention_forward
    GroupSumEdgeEmbedder.forward = _group_sum_forward


# ---------------------------------------------------------------- the wrapper's parts


def plato_lr_hypers(pet_model) -> Optional[dict]:
    """The model's ``plato_lr`` hypers with metatrain's defaults filled in, None if the
    model has no Plato-LR component."""
    frame_rope = bool(getattr(pet_model, "frame_rope", False))
    mesh = getattr(pet_model, "mesh", None)
    les = getattr(pet_model, "les", None)
    edge = any(type(g.edge_embedder).__name__ == "GroupSumEdgeEmbedder" for g in pet_model.gnn_layers)
    if not (frame_rope or mesh is not None or les is not None or edge):
        return None
    outer = getattr(pet_model, "outer_hypers", None) or {}
    h = dict(outer.get("plato_lr", {}) or {})
    hypers = getattr(pet_model, "hypers", None)
    if not h and isinstance(hypers, dict):
        h = dict(hypers.get("plato_lr", {}) or {})
    return dict(
        frames=str(h.get("frames", "identity")), frame_mode=str(h.get("frame_mode", "attention")),
        rope_init=str(h.get("rope_init", "gate")), rope=frame_rope, edge_group_sum=edge,
        mesh_coupling=mesh is not None, couple_at=int(getattr(pet_model, "couple_at", -1)),
        mesh_channels=int(mesh.coupling.c) if mesh is not None else 0,
        mesh_spacing=float(mesh.spacing) if mesh is not None else 0.0,
        mesh_sigma=float(mesh.sigma) if mesh is not None else 0.0,
        mesh_open_boundary=str(mesh.open_boundary) if mesh is not None else "none",
        n_quad=int(mesh.k_quad.numel()) if mesh is not None else 0,
        les=les is not None, num_charges=int(les.charge_head["node"][0][2].out_features) if les is not None else 0,
        les_sigma=float(les.sigma) if les is not None else 0.0, les_kcut=float(les.kcut) if les is not None else 0.0,
        total_charge_constraint=bool(les.total_charge_constraint) if les is not None else False,
    )


def fold_frame_linear(fl) -> (torch.Tensor, torch.Tensor):
    """(W_eff (c_out, c_in), b) of a one-frame FrameLinear, in float64."""
    I = torch.eye(3, dtype=torch.float64).reshape(1, 1, 9)
    with torch.no_grad():
        phi = fl.phi.to(torch.float64)
        k = phi(I).reshape(-1)  # (n_basis,)
        W = torch.einsum("b,boc->oc", k, fl.W.to(torch.float64))
        fl.phi.to(fl.W.dtype)
    return W, k


def unfold_mesh(fl, W_new: torch.Tensor, b_new: torch.Tensor) -> None:
    """Set a one-frame FrameLinear so that it folds to W_new (least-norm change of W)."""
    W_old, k = fold_frame_linear(fl)
    with torch.no_grad():
        dW = W_new.to(torch.float64) - W_old
        fl.W.add_((k[:, None, None] * dW[None] / (k @ k)).to(fl.W.dtype))
        if fl.b is not None:
            fl.b.copy_(b_new.to(fl.b.dtype))


class PlatoLRParts(torch.nn.Module):
    """The wrapper-side mesh and LES parts (registered as ``plato`` on PETFullModel; GGUF
    names ``plato_mesh_*`` and ``plato_les_*``)."""

    def __init__(self, pet_model, hypers: dict) -> None:
        super().__init__()
        self.h = hypers
        mesh, les = getattr(pet_model, "mesh", None), getattr(pet_model, "les", None)
        self.mesh_key = ""
        if mesh is not None:
            c = mesh.coupling
            if any(getattr(c, x).folded(1) is False for x in ("down", "up")):
                raise ValueError("plato_lr mesh FrameLinear not foldable")
            self.mesh_norm = c.norm
            self.mesh_filt = c.filt
            for name in ("down", "up"):
                fl = getattr(c, name)
                W, _ = fold_frame_linear(fl)
                lin = torch.nn.Linear(W.shape[1], W.shape[0], bias=fl.b is not None)
                with torch.no_grad():
                    lin.weight.copy_(W.to(fl.W.dtype))
                    if fl.b is not None:
                        lin.bias.copy_(fl.b)
                setattr(self, "mesh_" + name, lin)
            self.mesh_gate = mesh.mesh_gate
            self.register_buffer("mesh_k_quad", mesh.k_quad.clone())
            self.register_buffer("mesh_w_quad", mesh.w_quad.clone())
            self.mesh_key = f"mesh{id(mesh)}"
            _REGISTRY[self.mesh_key] = mesh
            self.__dict__["_mesh_module"] = mesh
        if les is not None:
            self.les_charge_head = les.charge_head
            self.les_gate = les.les_gate
            self.register_buffer("les_charge_unit", les.charge_unit.clone())

    def mesh_update(self, x: torch.Tensor) -> torch.Tensor:
        h = self.h
        v = self.mesh_down(self.mesh_norm(x))
        f0, f2 = self.mesh_filt[0], self.mesh_filt[2]
        m = torch.ops.mlipcpp.plato_mesh(v, f0.weight, f0.bias, f2.weight, f2.bias, self.mesh_k_quad,
                                         self.mesh_w_quad, float(h["mesh_sigma"]), float(h["mesh_spacing"]),
                                         int(h["mesh_open_boundary"] == "realspace"), self.mesh_key)
        return self.mesh_gate * self.mesh_up(m)

    def les_energy(self, nodes: List[torch.Tensor], edges: List[torch.Tensor], padding_mask: torch.Tensor,
                   cutoff_factors: torch.Tensor) -> torch.Tensor:
        """Gated per-atom LES energies (n,); padding_mask True = padded (the wrapper's
        convention)."""
        q = None
        for i, (nh, eh) in enumerate(zip(self.les_charge_head["node"], self.les_charge_head["edge"])):
            qe = eh(edges[i])
            qe = torch.where(padding_mask.unsqueeze(-1), torch.zeros_like(qe), qe)
            qi = nh(nodes[i]) + (qe * cutoff_factors.unsqueeze(-1)).sum(1)
            q = qi if q is None else q + qi
        h = self.h
        e = torch.ops.mlipcpp.plato_les(q, self.les_charge_unit, float(h["les_sigma"]), float(h["les_kcut"]),
                                        int(h["total_charge_constraint"]))
        return self.les_gate * e

    def sync_back(self) -> None:
        """Write the folded mesh projections back into the FrameLinears (GGUF ->
        checkpoint)."""
        mesh = self.__dict__.get("_mesh_module")
        if mesh is None:
            return
        for name in ("down", "up"):
            lin = getattr(self, "mesh_" + name)
            unfold_mesh(getattr(mesh.coupling, name), lin.weight.detach(), lin.bias.detach())

    def state_dict_names(self, outer_names: Dict[int, str]) -> Dict[str, str]:
        """GGUF names of the folded tensors -> the metatrain names they stand for."""
        out = {}
        if self.mesh_key:
            for name in ("down", "up"):
                out[f"plato_mesh_{name}_weight"] = f"backend.mesh.coupling.{name}.W"
                out[f"plato_mesh_{name}_bias"] = f"backend.mesh.coupling.{name}.b"
        return out


def gguf_metadata(writer, h: dict) -> None:
    """``pet.plato_lr.*`` keys (absent: a plain PET)."""
    from metatrain.experimental.plato_pet_lr.modules.frames import platonic_group
    writer.add_int32("pet.plato_lr", 1)
    writer.add_string("pet.plato_lr.frames", h["frames"])
    R = platonic_group(h["frames"]).to(torch.float64)
    writer.add_int32("pet.plato_lr.n_frames", int(R.shape[0]))
    writer.add_int32("pet.plato_lr.rope", int(h["rope"]))
    writer.add_string("pet.plato_lr.rope_init", h["rope_init"])
    writer.add_int32("pet.plato_lr.edge_group_sum", int(h["edge_group_sum"]))
    writer.add_int32("pet.plato_lr.mesh_coupling", int(h["mesh_coupling"]))
    if h["mesh_coupling"]:
        writer.add_int32("pet.plato_lr.couple_at", int(h["couple_at"]))
        writer.add_int32("pet.plato_lr.mesh_channels", int(h["mesh_channels"]))
        writer.add_float32("pet.plato_lr.mesh_spacing", float(h["mesh_spacing"]))
        writer.add_float32("pet.plato_lr.mesh_sigma", float(h["mesh_sigma"]))
        writer.add_string("pet.plato_lr.mesh_open_boundary", h["mesh_open_boundary"])
        writer.add_int32("pet.plato_lr.n_quad", int(h["n_quad"]))
    writer.add_int32("pet.plato_lr.les", int(h["les"]))
    if h["les"]:
        writer.add_int32("pet.plato_lr.num_charges", int(h["num_charges"]))
        writer.add_float32("pet.plato_lr.les_sigma", float(h["les_sigma"]))
        writer.add_float32("pet.plato_lr.les_kcut", float(h["les_kcut"]))
        writer.add_int32("pet.plato_lr.total_charge_constraint", int(h["total_charge_constraint"]))
