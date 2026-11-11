#include "train/tail_loss.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace mlipcpp::train {

TailSet TailSet::load(const std::string &path, bool mixed) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("--tail-data: cannot open " + path);
  TailSet t;
  std::map<std::string, int> cls_index, group_index;
  std::vector<std::map<int64_t, int64_t>> local;
  std::string line;
  std::getline(f, line); // header
  int ln = 1;
  while (std::getline(f, line)) {
    ln++;
    if (line.empty()) continue;
    std::istringstream in(line);
    std::string g, c;
    TailPair p;
    if (!(in >> g >> c >> p.k >> p.l >> p.de_dft >> p.de_ref >> p.d))
      throw std::runtime_error(path + ":" + std::to_string(ln) + ": expected group class k l dE_dft dE_ref d");
    auto ci = cls_index.emplace(c, int(t.classes.size()));
    if (ci.second) t.classes.push_back(c), t.by_class.emplace_back();
    auto gi = group_index.emplace(g, int(t.groups.size()));
    if (gi.second) {
      t.groups.push_back({g, ci.first->second, {}, {}});
      t.by_class[ci.first->second].push_back(gi.first->second);
      local.emplace_back();
    }
    TailGroup &grp = t.groups[gi.first->second];
    if (grp.cls != ci.first->second && !mixed) throw std::runtime_error(path + ": group " + g + " has two classes");
    p.cls = ci.first->second;
    auto &loc = local[gi.first->second];
    auto at = [&](int64_t s) {
      auto it = loc.emplace(s, int64_t(grp.ids.size()));
      if (it.second) grp.ids.push_back(s);
      return it.first->second;
    };
    p.k = at(p.k), p.l = at(p.l);
    grp.pairs.push_back(p);
  }
  if (t.groups.empty()) throw std::runtime_error("--tail-data: no pairs in " + path);
  return t;
}

size_t TailSet::n_pairs() const {
  size_t n = 0;
  for (const auto &g : groups) n += g.pairs.size();
  return n;
}

double TailLoss::scale(const TailPair &p) const { return std::max(std::abs(p.de_ref), floor); }

double TailLoss::residual(const TailPair &p, const std::vector<double> &E) const {
  return (E.at(p.k) - E.at(p.l) - p.de_dft) / scale(p);
}

double TailLoss::eval(const std::vector<TailPair> &pairs, const std::vector<double> &E, std::vector<double> *grad) const {
  if (grad) grad->assign(E.size(), 0.0);
  if (pairs.empty()) return 0.0;
  const double w = weight / double(pairs.size());
  double L = 0;
  for (const TailPair &p : pairs) {
    const double r = residual(p, E), a = std::abs(r);
    L += w * (a < delta ? 0.5 * r * r : delta * (a - 0.5 * delta));
    if (grad) {
      const double g = w * std::max(-delta, std::min(delta, r)) / scale(p);
      (*grad)[p.k] += g, (*grad)[p.l] -= g;
    }
  }
  return L;
}

} // namespace mlipcpp::train
