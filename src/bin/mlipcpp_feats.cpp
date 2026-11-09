/**
 * Per-atom backbone features of a pet-graph GGUF (the inputs of the energy head's last linear layers, i.e. the
 * last-layer features), for offline fits of feature-based heads (e.g. the lrc charge head).
 *
 *   mlipcpp-feats MODEL.gguf --list                       GIR nodes (id op name shape inputs)
 *   mlipcpp-feats MODEL.gguf IN.xyz OUT.bin ID[,ID...]     per frame: the captured node outputs, float32
 *   mlipcpp-feats MODEL.gguf IN.xyz OUT.bin --llf NODE,EDGE,FC,PAD   per frame: int64 n, d; then float32 [n][d]:
 *       the node features NODE [d_n, n], sum_j fc_ij (1 - pad_ij) EDGE_ij ([d_e, m, n]) and sum_j fc_ij (1 - pad_ij)
 *
 * OUT.bin (ID mode): per frame and node, the raw float32 tensor (ggml order) preceded by int64 ne[0..3]. Charge and
 * spin multiplicity come from the frame's charge= / spin_multiplicity= keys.
 */
#include "mlipcpp/io.h"
#include "runtime/graph_model.h"

#include <ggml-backend.h>

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace mlipcpp;

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "usage: mlipcpp-feats MODEL.gguf --list | IN.xyz OUT.bin ID[,ID...]\n";
    return 1;
  }
  try {
    runtime::GraphModel m;
    if (!m.load_from_gguf(argv[1])) throw std::runtime_error("load failed");
    const auto &g = m.interpreter().graph();
    if (std::string(argv[2]) == "--list") {
      for (const auto &n : g.nodes) {
        std::cout << n.id << " " << n.op << " " << n.name << " [";
        for (auto s : n.output_shape) std::cout << s << ",";
        std::cout << "]";
        for (const auto &i : n.inputs) std::cout << " " << i;
        std::cout << "\n";
      }
      return 0;
    }
    const bool llf = std::string(argv[4]) == "--llf";
    std::vector<int> ids;
    {
      std::stringstream ss(argv[llf ? 5 : 4]);
      std::string t;
      while (std::getline(ss, t, ',')) ids.push_back(std::stoi(t));
    }
    std::ifstream in(argv[2]);
    std::ofstream out(argv[3], std::ios::binary);
    int nf = 0;
    while (in.peek() != EOF) {
      std::streampos at = in.tellg();
      std::string line;
      if (!std::getline(in, line)) break;
      if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
      in.seekg(at);
      AtomicSystem sys = io::read_xyz(in);
      m.predict_with_capture(sys, [&](runtime::GraphInterpreter &interp) {
        const auto &no = interp.node_outputs();
        auto grab = [&](int id) {
          auto it = no.find(id);
          if (it == no.end()) throw std::runtime_error("node not built: " + std::to_string(id));
          ggml_tensor *t = it->second;
          if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) throw std::runtime_error("node not f32/contiguous");
          std::vector<float> buf(ggml_nelements(t));
          ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * 4);
          return std::make_pair(buf, std::array<int64_t, 4>{t->ne[0], t->ne[1], t->ne[2], t->ne[3]});
        };
        if (llf) {
          auto [hn, nn] = grab(ids[0]);
          auto [he, ne] = grab(ids[1]);
          auto [fc, nf] = grab(ids[2]);
          auto [pd, np] = grab(ids[3]);
          const int64_t n = nn[1], dn = nn[0], de = ne[0], m = ne[1];
          if (ne[2] != n || nf[0] * nf[1] * nf[2] != m * n || np[0] * np[1] * np[2] != m * n)
            throw std::runtime_error("llf: shapes");
          const int64_t d = dn + de + 1;
          std::vector<float> o(n * d, 0.0f);
          for (int64_t i = 0; i < n; i++) {
            float *r = &o[i * d];
            for (int64_t k = 0; k < dn; k++) r[k] = hn[i * dn + k];
            for (int64_t j = 0; j < m; j++) {
              const float w = fc[i * m + j] * (pd[i * m + j] > 0.5f ? 0.0f : 1.0f);
              if (w == 0.0f) continue;
              for (int64_t k = 0; k < de; k++) r[dn + k] += w * he[(i * m + j) * de + k];
              r[dn + de] += w;
            }
          }
          out.write(reinterpret_cast<const char *>(&n), 8);
          out.write(reinterpret_cast<const char *>(&d), 8);
          out.write(reinterpret_cast<const char *>(o.data()), o.size() * 4);
          return;
        }
        for (int id : ids) {
          auto it = no.find(id);
          if (it == no.end()) throw std::runtime_error("node not built: " + std::to_string(id));
          ggml_tensor *t = it->second;
          if (t->type != GGML_TYPE_F32) throw std::runtime_error("node not f32");
          int64_t ne[4] = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
          std::vector<float> buf(ggml_nelements(t));
          if (ggml_is_contiguous(t)) ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * 4);
          else throw std::runtime_error("node not contiguous: " + std::to_string(id));
          out.write(reinterpret_cast<char *>(ne), sizeof ne);
          out.write(reinterpret_cast<char *>(buf.data()), buf.size() * 4);
        }
      });
      nf++;
    }
    std::cout << nf << " frames\n";
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
