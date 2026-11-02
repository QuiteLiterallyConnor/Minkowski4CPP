// coarsen() / adaptive_pool() against an oracle of the original algorithm: map every merged
// leaf to its parent, sort all entries by key (stable), merge equal keys and reduce them in
// index order (index_add_ from zeros, then / count for Mean; amax for Max). On the CPU the
// result must match bit for bit (torch::equal); CUDA must match the CPU (atomics reorder sums).
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <random>
#include <set>
#include <vector>

#include "MinkowskiEngine/adaptive/adaptive.hpp"
#include "utils/test_framework.hpp"

using namespace minkowski::adaptive;

namespace {

constexpr uint32_t kDepth = 4;
constexpr double kBase = 8.0;

AdaptiveVoxelConfig config() {
    AdaptiveVoxelConfig c;
    c.base_voxel_size = {kBase, kBase, kBase};
    const double fine = kBase / double(1u << kDepth);
    c.min_voxel_size = {fine, fine, fine};
    c.max_depth = kDepth;
    return c;
}

struct K {
    uint32_t level;
    std::array<int64_t, 3> c;
    bool operator<(const K &o) const { return level != o.level ? level < o.level : c < o.c; }
    bool operator==(const K &o) const { return level == o.level && c == o.c; }
};

K parent(const K &k) { return {k.level - 1, {k.c[0] >> 1, k.c[1] >> 1, k.c[2] >> 1}}; }

// Non-overlapping random leaves: random finest cells in a box of `cells` level-0 cells
// (negative coordinates included), each lifted to a random level; overlaps are rejected.
std::vector<K> random_leaves(std::mt19937_64 &rng, int n, int cells, uint32_t min_level = 0) {
    const int64_t span = int64_t(cells) << kDepth;
    std::uniform_int_distribution<int64_t> pos(-span / 2, span / 2 - 1);
    std::uniform_int_distribution<uint32_t> lvl(min_level, kDepth);
    std::set<K> leaves, blocked; // blocked = leaves and all their ancestors
    for (int i = 0; i < n; ++i) {
        const uint32_t l = lvl(rng);
        const int s = int(kDepth - l);
        K k{l, {pos(rng) >> s, pos(rng) >> s, pos(rng) >> s}};
        if (blocked.count(k))
            continue; // k is a leaf or an ancestor of one
        bool ok = true;
        for (K a = k; a.level > 0 && ok;) {
            a = parent(a);
            ok = !leaves.count(a);
        }
        if (!ok)
            continue;
        leaves.insert(k);
        for (K a = k;; a = parent(a)) {
            blocked.insert(a);
            if (a.level == 0)
                break;
        }
    }
    return {leaves.begin(), leaves.end()};
}

AdaptiveSparseTensor make_tensor(const std::vector<K> &leaves, int64_t C, torch::Dtype dtype,
                                 std::mt19937_64 &rng, bool signed_zero_row = true) {
    const int64_t n = static_cast<int64_t>(leaves.size());
    auto coords = torch::empty({n, 3}, torch::kInt64);
    auto levels = torch::empty({n}, torch::kInt32);
    for (int64_t i = 0; i < n; ++i) {
        for (int d = 0; d < 3; ++d)
            coords[i][d] = leaves[i].c[d];
        levels[i] = static_cast<int32_t>(leaves[i].level);
    }
    torch::manual_seed(static_cast<uint64_t>(rng()));
    auto feats = torch::randn({n, C}, torch::TensorOptions().dtype(dtype));
    if (n > 0 && signed_zero_row)
        feats[0].fill_(-0.0); // -0.0 rows must survive pooling as zeros
    return AdaptiveSparseTensor::from_voxels(coords, levels, feats, config());
}

bool inside(const K &k, const AxisAlignedRegion &r) {
    const double s = kBase * std::ldexp(1.0, -static_cast<int>(k.level));
    for (int d = 0; d < 3; ++d) {
        const double lo = static_cast<double>(k.c[d]) * s, hi = lo + s;
        if (!(lo >= r.lo[d] && hi <= r.hi[d]))
            return false;
    }
    return true;
}

struct Oracle {
    std::vector<K> keys;
    at::Tensor feats;
};

// The original coarsen(): one sort of every entry per level, index_add_ / scatter_reduce.
Oracle oracle_coarsen(std::vector<K> keys, at::Tensor f, const AxisAlignedRegion &r,
                      uint32_t target, Reduction red) {
    for (int64_t level = kDepth; level > static_cast<int64_t>(target); --level) {
        struct E {
            K key;
            int64_t src;
        };
        std::vector<E> entries;
        bool changed = false;
        for (int64_t i = 0; i < static_cast<int64_t>(keys.size()); ++i) {
            const K &k = keys[i];
            if (k.level == level && level >= 1 && inside(parent(k), r)) {
                entries.push_back({parent(k), i});
                changed = true;
            } else {
                entries.push_back({k, i});
            }
        }
        if (!changed)
            continue;
        std::stable_sort(entries.begin(), entries.end(),
                         [](const E &a, const E &b) { return a.key < b.key; });
        std::vector<K> out;
        std::vector<int64_t> src, dst;
        for (const E &e : entries) {
            if (out.empty() || !(out.back() == e.key))
                out.push_back(e.key);
            src.push_back(e.src);
            dst.push_back(static_cast<int64_t>(out.size()) - 1);
        }
        const int64_t nn = static_cast<int64_t>(out.size()), M = static_cast<int64_t>(src.size());
        auto src_t = torch::tensor(src, torch::kInt64);
        auto dst_t = torch::tensor(dst, torch::kInt64);
        auto g = f.index_select(0, src_t);
        auto o = torch::zeros({nn, f.size(1)}, f.options());
        if (red == Reduction::Max) {
            o = torch::scatter_reduce(o, 0, dst_t.unsqueeze(1).expand({M, f.size(1)}), g, "amax",
                                      false);
        } else {
            o.index_add_(0, dst_t, g);
            if (red == Reduction::Mean) {
                std::vector<double> cnt(static_cast<size_t>(nn), 0.0);
                for (int64_t d : dst)
                    cnt[static_cast<size_t>(d)] += 1.0;
                for (double &c : cnt)
                    c = std::max(c, 1.0);
                o = o / torch::tensor(cnt, torch::kFloat64).to(f.scalar_type()).unsqueeze(1);
            }
        }
        keys = std::move(out);
        f = o;
    }
    return {keys, f};
}

std::vector<K> keys_of(const AdaptiveSparseTensor &t) {
    auto c = t.coordinates().cpu();
    auto l = t.levels().cpu();
    std::vector<K> k(static_cast<size_t>(t.size()));
    for (int64_t i = 0; i < t.size(); ++i)
        k[i] = {static_cast<uint32_t>(l[i].item<int32_t>()),
                {c[i][0].item<int64_t>(), c[i][1].item<int64_t>(), c[i][2].item<int64_t>()}};
    return k;
}

AxisAlignedRegion everything() { return AxisAlignedRegion::everything(3); }

AxisAlignedRegion random_box(std::mt19937_64 &rng, int cells) {
    std::uniform_real_distribution<double> u(-cells * kBase / 2, cells * kBase / 2);
    AxisAlignedRegion r{{0, 0, 0}, {0, 0, 0}};
    for (int d = 0; d < 3; ++d) {
        const double a = u(rng), b = u(rng);
        r.lo[d] = std::min(a, b) - kBase;
        r.hi[d] = std::max(a, b) + kBase;
    }
    return r;
}

const Reduction kReductions[] = {Reduction::Mean, Reduction::Sum, Reduction::Max};

} // namespace

TEST(coarsen_merge_matches_full_sort_oracle_bitwise_cpu) {
    std::mt19937_64 rng(20261102);
    int cases = 0;
    for (int trial = 0; trial < 24; ++trial) {
        const auto leaves = random_leaves(rng, 40 + 60 * (trial % 6), 2 + trial % 5);
        const torch::Dtype dtype = trial % 2 ? torch::kFloat64 : torch::kFloat32;
        const auto t0 = make_tensor(leaves, 1 + trial % 7, dtype, rng);
        for (Reduction red : kReductions)
            for (uint32_t target : {0u, 1u, 2u, 3u}) {
                const auto region = trial % 3 == 0 ? everything() : random_box(rng, 2 + trial % 5);
                auto t = t0;
                t.coarsen(region, target, red);
                const auto ref = oracle_coarsen(leaves, t0.features(), region, target, red);
                CHECK(keys_of(t) == ref.keys);
                CHECK(torch::equal(t.features(), ref.feats));
                CHECK(!t.has_overlapping_leaves());
                ++cases;
            }
    }
    CHECK_EQ(cases, 24 * 3 * 4);
}

TEST(coarsen_merge_adaptive_pool_matches_oracle_and_keeps_input) {
    std::mt19937_64 rng(7);
    const auto leaves = random_leaves(rng, 400, 4);
    const auto t = make_tensor(leaves, 5, torch::kFloat32, rng);
    const auto before = t.features().clone();
    for (Reduction red : kReductions)
        for (uint32_t target = 0; target <= kDepth; ++target) {
            auto p = adaptive_pool(t, target, red);
            const auto ref = oracle_coarsen(leaves, t.features(), everything(), target, red);
            CHECK(keys_of(p) == ref.keys);
            CHECK(torch::equal(p.features(), ref.feats));
        }
    CHECK(torch::equal(t.features(), before));
    CHECK(keys_of(t) == leaves);
}

TEST(coarsen_merge_edge_cases) {
    std::mt19937_64 rng(3);
    // Empty tensor.
    auto e = make_tensor({}, 4, torch::kFloat32, rng);
    e.coarsen(everything(), 0, Reduction::Mean);
    CHECK_EQ(e.size(), int64_t(0));
    // Only level-0 leaves: nothing to merge, features untouched (same tensor).
    const auto flat = make_tensor({{0, {0, 0, 0}}, {0, {3, 1, -2}}}, 2, torch::kFloat64, rng);
    auto f = flat;
    f.coarsen(everything(), 0, Reduction::Sum);
    CHECK(f.features().is_same(flat.features()));
    // A region that holds no complete parent: unchanged.
    std::vector<K> kids;
    for (int i = 0; i < 8; ++i)
        kids.push_back({1, {i & 1, (i >> 1) & 1, (i >> 2) & 1}});
    const auto t = make_tensor(kids, 3, torch::kFloat32, rng);
    auto u = t;
    u.coarsen(AxisAlignedRegion{{0, 0, 0}, {4, 4, 4}}, 0, Reduction::Mean);
    CHECK_EQ(u.size(), int64_t(8));
    CHECK(torch::equal(u.features(), t.features()));
    // Every leaf merges into one parent; -0.0 children give a +/-0 result equal to 0.
    auto w = t;
    w.coarsen(everything(), 0, Reduction::Mean);
    CHECK_EQ(w.size(), int64_t(1));
    CHECK_ALLCLOSE(w.features(), t.features().mean(0, true), 1e-6, 1e-7);
    // Parents interleaved with existing level-0 leaves on both sides.
    std::vector<K> mix = {{0, {-1, 0, 0}}, {0, {2, 0, 0}}};
    for (const K &k : kids)
        mix.push_back({1, {k.c[0] + 2, k.c[1], k.c[2]}}); // children of (1, 0, 0)
    std::sort(mix.begin(), mix.end());
    const auto m = make_tensor(mix, 2, torch::kFloat64, rng);
    for (Reduction red : kReductions) {
        auto c = m;
        c.coarsen(everything(), 0, red);
        const auto ref = oracle_coarsen(mix, m.features(), everything(), 0, red);
        CHECK(keys_of(c) == ref.keys);
        CHECK(torch::equal(c.features(), ref.feats));
        CHECK_EQ(c.size(), int64_t(3));
    }
}

// No exact zeros here: for a maximum of exactly 0, torch's scatter_reduce(amax,
// include_self=false) backward also counts the zero-initialised output as a maximum and
// halves the gradient, which the oracle (and the old code) did at every step for every row.
TEST(coarsen_merge_gradients_match_oracle) {
    std::mt19937_64 rng(11);
    for (int trial = 0; trial < 6; ++trial) {
        const auto leaves = random_leaves(rng, 150, 3);
        const auto t = make_tensor(leaves, 3, torch::kFloat64, rng, /*signed_zero_row=*/false);
        const auto region = trial % 2 ? everything() : random_box(rng, 3);
        for (Reduction red : kReductions) {
            const uint32_t target = static_cast<uint32_t>(trial % 3);
            auto x = t.features().clone().requires_grad_(true);
            auto c = t.with_features(x);
            c.coarsen(region, target, red);
            torch::manual_seed(static_cast<uint64_t>(trial));
            auto probe = torch::randn_like(c.features());
            (c.features() * probe).sum().backward();
            auto xr = t.features().clone().requires_grad_(true);
            const auto ref = oracle_coarsen(leaves, xr, region, target, red);
            (ref.feats * probe).sum().backward();
            CHECK_ALLCLOSE(x.grad(), xr.grad(), 0, 1e-12);
        }
    }
}

TEST(coarsen_merge_cuda_matches_cpu) {
    if (!torch::cuda::is_available())
        throw mink_test::Skip("CUDA not available");
    std::mt19937_64 rng(99);
    for (int trial = 0; trial < 8; ++trial) {
        const auto leaves = random_leaves(rng, 300, 4);
        const torch::Dtype dtype = trial % 2 ? torch::kFloat64 : torch::kFloat32;
        const auto t = make_tensor(leaves, 6, dtype, rng);
        const auto region = trial % 2 ? everything() : random_box(rng, 4);
        for (Reduction red : kReductions) {
            const uint32_t target = static_cast<uint32_t>(trial % 3);
            auto xc = t.features().clone().requires_grad_(true);
            auto xg = t.features().to(torch::kCUDA).requires_grad_(true);
            auto c = t.with_features(xc);
            auto g = t.with_features(xg);
            c.coarsen(region, target, red);
            g.coarsen(region, target, red);
            CHECK(keys_of(c) == keys_of(g));
            const double tol = dtype == torch::kFloat32 ? 1e-5 : 1e-12;
            CHECK_ALLCLOSE(g.features().cpu(), c.features(), tol, tol);
            auto probe = torch::randn_like(c.features());
            (c.features() * probe).sum().backward();
            (g.features() * probe.to(torch::kCUDA)).sum().backward();
            CHECK_ALLCLOSE(xg.grad().cpu(), xc.grad(), tol, tol);
        }
    }
}
