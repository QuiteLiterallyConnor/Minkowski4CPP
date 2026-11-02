// Equivalence of the tree-descent Topology::for_each_overlap with the original grid scan
// (Topology::for_each_overlap_scan, kept as the oracle) on random adaptive trees, and of the
// OverlapWeighted kernel maps and Multilinear interpolation built on top of it.
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "MinkowskiEngine/adaptive/adaptive.hpp"
#include "MinkowskiEngine/adaptive/adaptive_topology.hpp"
#include "utils/test_framework.hpp"

using namespace minkowski::adaptive;
namespace adt = minkowski::adaptive::detail;
using adt::DVec;
using adt::Key;
using adt::Topology;

namespace {

using Hits = std::vector<std::pair<int64_t, double>>;

struct TreeSpec {
    int D;
    uint32_t depth;
    uint32_t root_level;  // coarsest level a leaf can have
    int root_extent;      // roots cover [-root_extent, root_extent)^D at root_level
    double p_refine;      // probability that a cell is split further
    double p_drop;        // probability that a leaf is left out (sparse trees)
    bool dyadic;          // base size 8 and origin 0, else irregular sizes and origin
};

std::shared_ptr<const Topology> random_topology(const TreeSpec& spec, std::mt19937_64& rng) {
    AdaptiveVoxelConfig cfg;
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    for (int d = 0; d < spec.D; ++d) {
        cfg.base_voxel_size.push_back(spec.dyadic ? 8.0 : 0.3 + 1.7 * u01(rng));
        cfg.origin.push_back(spec.dyadic ? 0.0 : -5.0 + 10.0 * u01(rng));
    }
    cfg.max_depth = spec.depth;
    std::vector<Key> leaves;
    std::vector<Key> stack;
    Key root{};
    root.level = spec.root_level;
    std::vector<int64_t> c(spec.D, -spec.root_extent);
    while (true) {
        for (int d = 0; d < spec.D; ++d) root.coordinates[d] = c[d];
        stack.push_back(root);
        int d = spec.D - 1;
        for (; d >= 0; --d) {
            if (++c[d] < spec.root_extent) break;
            c[d] = -spec.root_extent;
        }
        if (d < 0) break;
    }
    std::vector<Key> children;
    while (!stack.empty()) {
        Key k = stack.back();
        stack.pop_back();
        if (k.level < spec.depth && u01(rng) < spec.p_refine) {
            children.clear();
            adt::child_keys(k, spec.D, children);
            stack.insert(stack.end(), children.begin(), children.end());
        } else if (u01(rng) >= spec.p_drop) {
            leaves.push_back(k);
        }
    }
    std::sort(leaves.begin(), leaves.end(), adt::key_less);
    return Topology::make(cfg, std::move(leaves));
}

Hits run_descent(const Topology& t, const DVec& lo, const DVec& hi) {
    Hits h;
    t.for_each_overlap(lo, hi, [&](int64_t j, double ov) { h.emplace_back(j, ov); });
    return h;
}

Hits run_scan(const Topology& t, const DVec& lo, const DVec& hi) {
    Hits h;
    t.for_each_overlap_scan(lo, hi, [&](int64_t j, double ov) { h.emplace_back(j, ov); });
    return h;
}

// Same leaves, same order, bit-identical overlap volumes.
void check_same(const Topology& t, const DVec& lo, const DVec& hi, int64_t& total_hits) {
    const Hits a = run_descent(t, lo, hi);
    const Hits b = run_scan(t, lo, hi);
    if (a.size() != b.size())
        FAIL_AT("hit count differs: descent " << a.size() << " vs scan " << b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].first != b[i].first || a[i].second != b[i].second)
            FAIL_AT("hit " << i << " differs: descent (" << a[i].first << ", " << a[i].second
                           << ") vs scan (" << b[i].first << ", " << b[i].second << ")");
        if (i > 0) CHECK(a[i].first > a[i - 1].first);
    }
    total_hits += static_cast<int64_t>(a.size());
}

// Boxes of every kind the callers produce plus adversarial ones.
void check_random_boxes(const Topology& t, std::mt19937_64& rng, int boxes) {
    const int D = t.D;
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::uniform_int_distribution<int64_t> pick(0, t.size() - 1);
    std::uniform_int_distribution<int> off(-1, 1);
    std::uniform_int_distribution<uint32_t> lvl(0, t.max_depth);
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    int64_t total = 0;
    for (int n = 0; n < boxes; ++n) {
        const Key& k = t.keys[pick(rng)];
        DVec klo{}, khi{};
        t.bounds(k, klo, khi);
        DVec lo{}, hi{};
        const int kind = n % 8;
        const DVec s = t.size_at(kind == 0 ? k.level : lvl(rng));
        for (int d = 0; d < D; ++d) {
            const double c = klo[d] + 0.5 * (khi[d] - klo[d]);
            switch (kind) {
            case 0:  // kernel tap box of a leaf (build_pairs, output == input)
            case 1:  // tap box with the cell size of another level (strided outputs)
            {
                const double t0 = c + off(rng) * s[d];
                lo[d] = t0 - 0.5 * s[d];
                hi[d] = t0 + 0.5 * s[d];
                break;
            }
            case 2: {  // Multilinear interpolation box around a random point
                const double q = klo[d] + (khi[d] - klo[d]) * (2.0 * u01(rng) - 0.5);
                lo[d] = q - 0.5 * s[d];
                hi[d] = q + 0.5 * s[d];
                break;
            }
            case 3: {  // arbitrary box spanning a few cells
                const double a = klo[d] + (u01(rng) - 0.5) * 3.0 * s[d];
                lo[d] = a;
                hi[d] = a + u01(rng) * 3.0 * s[d];
                break;
            }
            case 4: {  // edges a hair off grid lines (tolerance boundaries)
                static const double eps[] = {0.0, 1e-12, 5e-10, 1e-9, 2e-9, 1e-7};
                const double e1 = eps[n / 8 % 6] * s[d] * (u01(rng) < 0.5 ? -1 : 1);
                const double e2 = eps[(n / 48) % 6] * s[d] * (u01(rng) < 0.5 ? -1 : 1);
                lo[d] = klo[d] + e1;
                hi[d] = khi[d] + off(rng) * s[d] + e2;
                break;
            }
            case 5: {  // degenerate: zero width or inverted
                lo[d] = c;
                hi[d] = u01(rng) < 0.5 ? c : c - s[d];
                break;
            }
            case 6: {  // far away or huge
                const double big = u01(rng) < 0.5 ? 1e3 : 1e25;
                lo[d] = u01(rng) < 0.5 ? c + big : -big;
                hi[d] = lo[d] < 0 ? big : lo[d] + s[d];
                break;
            }
            default: {  // non-finite
                const int w = static_cast<int>(u01(rng) * 3);
                lo[d] = w == 0 ? -inf : (w == 1 ? nan : klo[d]);
                hi[d] = w == 2 ? inf : khi[d];
                break;
            }
            }
        }
        // Huge boxes over fine levels would make the scan oracle itself enumerate forever.
        bool scannable = true;
        for (uint32_t l : t.present_levels) {
            const DVec sl = t.size_at(l);
            double cells = 1.0;
            for (int d = 0; d < D; ++d) {
                const double w = (hi[d] - lo[d]) / sl[d];
                if (std::isfinite(w) && w > 0) cells *= w + 2.0;
            }
            if (cells > 2e5) scannable = false;
        }
        if (!scannable) {
            // Beyond +-4e18 cells both implementations return nothing at once.
            bool representable = true;
            for (int d = 0; d < D; ++d)
                representable = representable && std::fabs(lo[d] / t.base[d]) <= 4e18 &&
                                std::fabs(hi[d] / t.base[d]) <= 4e18;
            if (!representable) CHECK(run_descent(t, lo, hi).empty());
            continue;
        }
        check_same(t, lo, hi, total);
    }
    CHECK(total > 0);
}

}  // namespace

TEST(overlap_descent_matches_grid_scan_on_random_trees) {
    std::mt19937_64 rng(20261101);
    const std::vector<TreeSpec> specs = {
        {2, 2, 0, 3, 0.5, 0.0, true},   {2, 4, 0, 3, 0.5, 0.2, false},
        {2, 6, 1, 2, 0.55, 0.1, false}, {3, 2, 0, 2, 0.4, 0.0, true},
        {3, 4, 0, 2, 0.3, 0.1, false},  {3, 6, 0, 1, 0.3, 0.0, true},
        {3, 6, 2, 1, 0.28, 0.3, false}, {4, 2, 0, 1, 0.3, 0.0, true},
        {4, 4, 0, 1, 0.15, 0.1, false}, {4, 6, 1, 1, 0.13, 0.0, false},
    };
    for (const auto& spec : specs) {
        for (int rep = 0; rep < 2; ++rep) {
            auto t = random_topology(spec, rng);
            if (t->size() == 0) continue;
            check_random_boxes(*t, rng, 2000);
        }
    }
}

TEST(overlap_descent_handles_empty_and_single_leaf_topologies) {
    AdaptiveVoxelConfig cfg;
    cfg.base_voxel_size = {1.0, 1.0, 1.0};
    cfg.max_depth = 3;
    auto empty = Topology::make(cfg, {});
    DVec lo{0, 0, 0, 0}, hi{1, 1, 1, 0};
    CHECK(run_descent(*empty, lo, hi).empty());

    Key k{};
    k.level = 3;
    k.coordinates = {5, -3, 2, 0};
    auto one = Topology::make(cfg, {k});
    DVec a{}, b{};
    one->bounds(k, a, b);
    auto h = run_descent(*one, a, b);
    CHECK_EQ(h.size(), size_t(1));
    CHECK_EQ(h[0].first, int64_t(0));
    CHECK_NEAR(h[0].second, 1.0 / 512.0, 1e-15);
    int64_t total = 0;
    check_same(*one, DVec{-10, -10, -10, 0}, DVec{10, 10, 10, 0}, total);
}

namespace {

using MapKey = std::tuple<int64_t, int64_t, int64_t>;  // kernel slot, out, in

// OverlapWeighted pairs exactly as build_pairs makes them, from the scan oracle.
std::vector<std::pair<MapKey, double>> oracle_overlap_map(const Topology& t, int K) {
    const int D = t.D;
    const int r = (K - 1) / 2;
    int64_t nk = 1;
    for (int d = 0; d < D; ++d) nk *= K;
    std::vector<std::vector<std::pair<MapKey, double>>> per_k(nk);
    for (int64_t o = 0; o < t.size(); ++o) {
        DVec lo, hi;
        t.bounds(t.keys[o], lo, hi);
        const DVec s = t.size_at(t.keys[o].level);
        double Vo = 1.0;
        DVec c{};
        for (int d = 0; d < D; ++d) {
            Vo *= s[d];
            c[d] = lo[d] + 0.5 * s[d];
        }
        for (int64_t k = 0; k < nk; ++k) {
            int64_t rem = k;
            DVec blo{}, bhi{};
            for (int d = D - 1; d >= 0; --d) {
                const int offd = static_cast<int>(rem % K) - r;
                rem /= K;
                const double tt = c[d] + offd * s[d];
                blo[d] = tt - 0.5 * s[d];
                bhi[d] = tt + 0.5 * s[d];
            }
            t.for_each_overlap_scan(blo, bhi, [&](int64_t j, double ov) {
                per_k[k].push_back({MapKey{k, o, j}, ov / Vo});
            });
        }
    }
    std::vector<std::pair<MapKey, double>> all;
    for (auto& v : per_k) all.insert(all.end(), v.begin(), v.end());
    return all;
}

}  // namespace

TEST(overlap_descent_kernel_map_is_identical_to_scan_oracle) {
    std::mt19937_64 rng(7);
    for (const TreeSpec& spec : {TreeSpec{2, 5, 0, 2, 0.5, 0.1, false},
                                 TreeSpec{3, 4, 0, 1, 0.35, 0.1, false},
                                 TreeSpec{4, 3, 0, 1, 0.2, 0.0, true}}) {
        auto topo = random_topology(spec, rng);
        auto x = AdaptiveSparseTensor::from_topology(
            topo, torch::randn({topo->size(), 2}, torch::kFloat64));
        AdaptiveKernelConfig kc;
        kc.mode = AdaptiveConvMode::OverlapWeighted;
        kc.kernel_size = spec.D == 4 ? 3 : 5;
        auto m = build_adaptive_kernel_map(x, kc);
        auto expected = oracle_overlap_map(*topo, kc.kernel_size);
        CHECK_EQ(m.in_index.numel(), static_cast<int64_t>(expected.size()));
        auto in = m.in_index.contiguous();
        auto out = m.out_index.contiguous();
        auto ki = m.kernel_index.contiguous();
        auto w = m.weight.contiguous();
        const int64_t* inp = in.data_ptr<int64_t>();
        const int64_t* outp = out.data_ptr<int64_t>();
        const int64_t* kp = ki.data_ptr<int64_t>();
        const double* wp = w.data_ptr<double>();
        for (size_t n = 0; n < expected.size(); ++n) {
            const auto& [key, wt] = expected[n];
            if (kp[n] != std::get<0>(key) || outp[n] != std::get<1>(key) ||
                inp[n] != std::get<2>(key) || wp[n] != wt)
                FAIL_AT("kernel map entry " << n << " differs");
        }
    }
}

TEST(overlap_descent_multilinear_interpolation_is_identical_to_scan_oracle) {
    std::mt19937_64 rng(11);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    for (const TreeSpec& spec : {TreeSpec{2, 6, 0, 2, 0.5, 0.0, false},
                                 TreeSpec{3, 5, 0, 1, 0.33, 0.05, false}}) {
        auto topo = random_topology(spec, rng);
        const int D = topo->D;
        auto feats = torch::randn({topo->size(), 3}, torch::kFloat64);
        auto x = AdaptiveSparseTensor::from_topology(topo, feats);
        auto [blo, bhi] = x.bounds();
        auto lo = std::get<0>(blo.min(0)), hi = std::get<0>(bhi.max(0));
        const int64_t M = 3000;
        auto pts = torch::empty({M, D}, torch::kFloat64);
        for (int64_t m = 0; m < M; ++m)
            for (int d = 0; d < D; ++d) {
                const double a = lo[d].item<double>(), b = hi[d].item<double>();
                pts[m][d] = a + (b - a) * u01(rng);
            }
        auto got = interpolate(x, pts, InterpMode::Multilinear);
        // Oracle: interpolate()'s Multilinear weights from the scan.
        auto expected = torch::zeros({M, 3}, torch::kFloat64);
        const double* p = pts.data_ptr<double>();
        for (int64_t m = 0; m < M; ++m) {
            DVec q{};
            for (int d = 0; d < D; ++d) q[d] = p[m * D + d];
            const int64_t j = topo->find_containing(q);
            if (j < 0) continue;
            const DVec s = topo->size_at(topo->keys[j].level);
            DVec a{}, b{};
            for (int d = 0; d < D; ++d) {
                a[d] = q[d] - 0.5 * s[d];
                b[d] = q[d] + 0.5 * s[d];
            }
            Hits h = run_scan(*topo, a, b);
            double total = 0.0;
            for (auto& e : h) total += e.second;
            for (auto& e : h) expected[m] += feats[e.first] * (e.second / total);
        }
        CHECK_ALLCLOSE(got, expected, 0, 1e-12);
    }
}
