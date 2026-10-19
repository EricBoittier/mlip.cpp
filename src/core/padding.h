#pragma once
// The contents of PET's padded neighbour slots (padding_mask = 1).
//
// Padded slots are masked by a finite attention bias, log(clamp(0, 1e-15)) =
// -34.5, not by -inf, so their tokens still enter the softmax with a tiny
// weight. metatrain fills them with a copy of a real edge (edge 0 of the
// batch), whose token is ordinary and therefore negligible. A zero edge vector
// is not: its distance is ~0 and its token large enough to outweigh the bias
// (up to ~20 meV/atom on MPtrj test structures for a trained model), and the
// in-graph sqrt(0) has no finite derivative.
//
// So every packer (GraphModel, mlipcpp-train, the MD engine) fills padded
// slots with one fixed, ordinary edge: species index 0 and the vector
// (0, 0, cutoff), at the cutoff so the in-graph cutoff factor is 0 as well.
// With it, metatrain's energies are reproduced to 1e-10 meV/atom (the same as
// copying any real edge), independent of edge order, and an atom without
// neighbours gets the same single padded slot.

#if defined(__CUDACC__)
#define MLIPCPP_PAD_HD __host__ __device__
#else
#define MLIPCPP_PAD_HD
#endif

namespace mlipcpp::padding {

constexpr int kSpecies = 0;

MLIPCPP_PAD_HD inline void edge(float cutoff, float out[3]) {
  out[0] = 0.0f, out[1] = 0.0f, out[2] = cutoff;
}

} // namespace mlipcpp::padding
