// Helpers shared by the ContinuousMultilinear interpolation tests and adaptive_interp_eval:
// random adaptive trees, a brute-force tent-partition oracle and leaf-boundary crossings
// along lines (for continuity measurements).
#ifndef MINK_TESTS_ADAPTIVE_INTERP_EVAL_UTIL_HPP
#define MINK_TESTS_ADAPTIVE_INTERP_EVAL_UTIL_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include "MinkowskiEngine/adaptive/adaptive.hpp"
#include "MinkowskiEngine/adaptive/adaptive_topology.hpp"

namespace interp_eval {

namespace adt = minkowski::adaptive::detail;
using adt::DVec;
using adt::Key;
using adt::Topology;

// Level-0 roots [-extent, extent)^D, each refined with probability p_refine per level down to
// `depth`; final cells are dropped with probability p_drop. Base size and origin are random
// (anisotropic) unless `iso` is set (base 1, origin 0).
inline std::shared_ptr<const Topology> random_tree(int D, uint32_t depth, int extent,
                                                   double p_refine, double p_drop,
                                                   std::mt19937_64& rng, bool iso = false) {
    minkowski::adaptive::AdaptiveVoxelConfig cfg;
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    for (int d = 0; d < D; ++d) {
        cfg.base_voxel_size.push_back(iso ? 1.0 : 0.5 + u01(rng));
        cfg.origin.push_back(iso ? 0.0 : -2.0 + 4.0 * u01(rng));
    }
    cfg.max_depth = depth;
    std::vector<Key> leaves, stack, children;
    std::vector<int64_t> c(D, -extent);
    while (true) {
        Key root{};
        for (int d = 0; d < D; ++d) root.coordinates[d] = c[d];
        stack.push_back(root);
        int d = D - 1;
        for (; d >= 0; --d) {
            if (++c[d] < extent) break;
            c[d] = -extent;
        }
        if (d < 0) break;
    }
    while (!stack.empty()) {
        Key k = stack.back();
        stack.pop_back();
        if (k.level < depth && u01(rng) < p_refine) {
            children.clear();
            adt::child_keys(k, D, children);
            stack.insert(stack.end(), children.begin(), children.end());
        } else if (u01(rng) >= p_drop) {
            leaves.push_back(k);
        }
    }
    std::sort(leaves.begin(), leaves.end(), adt::key_less);
    return Topology::make(cfg, std::move(leaves));
}

inline DVec leaf_center(const Topology& t, int64_t i) {
    DVec lo{}, hi{};
    t.bounds(t.keys[i], lo, hi);
    DVec c{};
    for (int d = 0; d < t.D; ++d) c[d] = 0.5 * (lo[d] + hi[d]);
    return c;
}

// Leaf containing q by a linear scan of every leaf (half-open boxes), or -1.
inline int64_t brute_containing(const Topology& t, const DVec& q) {
    for (int64_t i = 0; i < t.size(); ++i) {
        DVec lo{}, hi{};
        t.bounds(t.keys[i], lo, hi);
        bool in = true;
        for (int d = 0; d < t.D && in; ++d) in = q[d] >= lo[d] && q[d] < hi[d];
        if (in) return i;
    }
    return -1;
}

// Tent weight of leaf i at q: prod_d max(0, 1 - |q_d - c_id| / s_id).
inline double tent(const Topology& t, int64_t i, const DVec& q) {
    const DVec s = t.size_at(t.keys[i].level);
    const DVec c = leaf_center(t, i);
    double phi = 1.0;
    for (int d = 0; d < t.D; ++d) phi *= std::max(0.0, 1.0 - std::fabs(q[d] - c[d]) / s[d]);
    return phi;
}

// Brute-force ContinuousMultilinear weights over all leaves: (leaf, normalized weight) pairs,
// empty when q lies outside every leaf.
inline std::vector<std::pair<int64_t, double>> brute_tent_weights(const Topology& t,
                                                                  const DVec& q) {
    std::vector<std::pair<int64_t, double>> w;
    if (brute_containing(t, q) < 0) return w;
    double total = 0.0;
    for (int64_t i = 0; i < t.size(); ++i) {
        const double phi = tent(t, i, q);
        if (phi > 0.0) {
            w.emplace_back(i, phi);
            total += phi;
        }
    }
    for (auto& e : w) e.second /= total;
    return w;
}

// A pair of points 2 * delta apart that straddle the boundary between two different leaves
// (a, b: their indices) on a line.
struct Crossing {
    DVec before{}, after{};
    int64_t a = -1, b = -1;
    bool level_change = false;
};

// Walks the segment p0 -> p1 leaf by leaf: from the leaf containing the current point, jumps
// to the parameter where the line leaves that leaf's box and records the points delta (in
// space units) before and after it. Gaps outside every leaf are crossed in steps of `gap_step`
// (fraction of the segment).
inline std::vector<Crossing> line_crossings(const Topology& t, const DVec& p0, const DVec& p1,
                                            double delta, double gap_step = 1e-3) {
    std::vector<Crossing> out;
    DVec dir{};
    double len = 0.0;
    for (int d = 0; d < t.D; ++d) {
        dir[d] = p1[d] - p0[d];
        len += dir[d] * dir[d];
    }
    len = std::sqrt(len);
    if (!(len > 0.0)) return out;
    const double dt = delta / len;
    auto at = [&](double s) {
        DVec x{};
        for (int d = 0; d < t.D; ++d) x[d] = p0[d] + s * dir[d];
        return x;
    };
    double s = 0.0;
    for (int guard = 0; s < 1.0 && guard < 1000000; ++guard) {
        const int64_t j = t.find_containing(at(s));
        if (j < 0) {
            s += gap_step;
            continue;
        }
        DVec lo{}, hi{};
        t.bounds(t.keys[j], lo, hi);
        const DVec x = at(s);
        double s_exit = std::numeric_limits<double>::infinity();
        for (int d = 0; d < t.D; ++d) {
            if (dir[d] > 0) s_exit = std::min(s_exit, s + (hi[d] - x[d]) / dir[d]);
            else if (dir[d] < 0) s_exit = std::min(s_exit, s + (lo[d] - x[d]) / dir[d]);
        }
        if (!(s_exit + dt < 1.0)) break;
        Crossing c;
        c.before = at(s_exit - dt);
        c.after = at(s_exit + dt);
        c.a = t.find_containing(c.before);
        c.b = t.find_containing(c.after);
        if (c.a >= 0 && c.b >= 0 && c.a != c.b) {
            c.level_change = t.keys[c.a].level != t.keys[c.b].level;
            out.push_back(c);
        }
        s = std::max(s_exit + dt, s + 1e-12);
    }
    return out;
}

// [n, D] float64 tensor of points.
inline at::Tensor to_tensor(const std::vector<DVec>& pts, int D) {
    auto out = torch::empty({static_cast<int64_t>(pts.size()), D}, torch::kFloat64);
    double* p = out.data_ptr<double>();
    for (size_t i = 0; i < pts.size(); ++i)
        for (int d = 0; d < D; ++d) p[i * D + d] = pts[i][d];
    return out;
}

// Uniform random points in the bounding box of the leaves, enlarged by `pad` leaf-0 sizes.
inline std::vector<DVec> random_points(const Topology& t, int64_t n, std::mt19937_64& rng,
                                       double pad = 0.0) {
    DVec lo{}, hi{};
    for (int d = 0; d < t.D; ++d) {
        lo[d] = std::numeric_limits<double>::infinity();
        hi[d] = -lo[d];
    }
    for (int64_t i = 0; i < t.size(); ++i) {
        DVec a{}, b{};
        t.bounds(t.keys[i], a, b);
        for (int d = 0; d < t.D; ++d) {
            lo[d] = std::min(lo[d], a[d]);
            hi[d] = std::max(hi[d], b[d]);
        }
    }
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::vector<DVec> pts(static_cast<size_t>(n));
    for (auto& q : pts)
        for (int d = 0; d < t.D; ++d) {
            const double a = lo[d] - pad * t.base[d], b = hi[d] + pad * t.base[d];
            q[d] = a + (b - a) * u01(rng);
        }
    return pts;
}

}  // namespace interp_eval

#endif
