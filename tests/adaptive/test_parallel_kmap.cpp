// Multi-threaded kernel-map construction, the node-table find_containing and the hashed
// derive_output_topology must reproduce the original serial algorithms exactly: same pairs in
// the same order, same output leaves, same containing leaf for every point.
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <tuple>
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

// A tree refined at random down to `depth`, with some leaves dropped (sparse).
std::shared_ptr<const Topology> random_tree(int D, uint32_t depth, int extent, double p_refine,
                                            double p_drop, std::mt19937_64& rng) {
    AdaptiveVoxelConfig cfg;
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    for (int d = 0; d < D; ++d) {
        cfg.base_voxel_size.push_back(0.5 + u01(rng));
        cfg.origin.push_back(-2.0 + 4.0 * u01(rng));
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

// The original find_containing: probe every present level with an exact-key lookup.
int64_t oracle_containing(const Topology& t, const std::map<std::vector<int64_t>, int64_t>& idx,
                          const DVec& p) {
    for (uint32_t l : t.present_levels) {
        Key k;
        if (!t.locate(p, l, k)) return -1;
        std::vector<int64_t> key{static_cast<int64_t>(l)};
        for (int d = 0; d < t.D; ++d) key.push_back(k.coordinates[d]);
        auto it = idx.find(key);
        if (it != idx.end()) return it->second;
    }
    return -1;
}

// CenterSampled / VolumeWeighted taps: the leaf containing the centre of output-level cell
// `cell`, located on the integer grid at every present level (ancestor at coarser levels,
// the child whose lower corner is the centre at finer ones).
int64_t oracle_containing_center(const Topology& t,
                                 const std::map<std::vector<int64_t>, int64_t>& idx,
                                 const Key& cell) {
    for (uint32_t l : t.present_levels) {
        std::vector<int64_t> key{static_cast<int64_t>(l)};
        for (int d = 0; d < t.D; ++d) {
            const int64_t c = cell.coordinates[d];
            if (l <= cell.level) {
                int64_t a = c;
                for (uint32_t i = l; i < cell.level; ++i) a = a >= 0 ? a / 2 : -((-a + 1) / 2);
                key.push_back(a);
            } else {
                const int64_t m = int64_t(1) << (l - cell.level);
                key.push_back(c * m + m / 2);
            }
        }
        auto it = idx.find(key);
        if (it != idx.end()) return it->second;
    }
    return -1;
}

// True if p lies in leaf j's closed box, widened by a relative 1e-9 (rounding of t).
bool closed_contains(const Topology& t, int64_t j, const DVec& p) {
    DVec lo, hi;
    t.bounds(t.keys[j], lo, hi);
    for (int d = 0; d < t.D; ++d) {
        const double e = 1e-9 * (hi[d] - lo[d]);
        if (p[d] < lo[d] - e || p[d] > hi[d] + e) return false;
    }
    return true;
}

std::map<std::vector<int64_t>, int64_t> key_index(const Topology& t) {
    std::map<std::vector<int64_t>, int64_t> idx;
    for (int64_t i = 0; i < t.size(); ++i) {
        std::vector<int64_t> key{static_cast<int64_t>(t.keys[i].level)};
        for (int d = 0; d < t.D; ++d) key.push_back(t.keys[i].coordinates[d]);
        idx[key] = i;
    }
    return idx;
}

// The original derive_output_topology (std::set, every level checked).
std::vector<Key> oracle_output_keys(const Topology& in, int shift) {
    std::set<Key, bool (*)(const Key&, const Key&)> keys(adt::key_less);
    for (const Key& k : in.keys) {
        const int64_t l = std::max<int64_t>(static_cast<int64_t>(k.level) + shift, 0);
        keys.insert(adt::ancestor_at(k, static_cast<uint32_t>(l)));
    }
    std::vector<Key> kept;
    for (const Key& k : keys) {
        bool covered = false;
        for (uint32_t l = 0; l < k.level && !covered; ++l)
            covered = keys.count(adt::ancestor_at(k, l)) > 0;
        if (!covered) kept.push_back(k);
    }
    return kept;
}

// The original serial build_pairs, on top of the oracles above.
struct Pairs {
    std::vector<int64_t> in, out, k;
    std::vector<double> w;
    std::vector<int64_t> offsets;
};

// `float_ties` (optional) counts CenterSampled/VolumeWeighted taps where the integer tap
// differs from the float point location of the original build_pairs; `bad_ties` those where
// the two answers are not both leaves touching the tap point (a real disagreement).
Pairs oracle_pairs(const Topology& out, const Topology& in, const AdaptiveKernelConfig& kc,
                   int64_t* float_ties = nullptr, int64_t* bad_ties = nullptr) {
    const int D = out.D, K = kc.kernel_size, r = (K - 1) / 2;
    int64_t nk = 1;
    for (int d = 0; d < D; ++d) nk *= K;
    const auto idx = key_index(in);
    std::vector<std::vector<std::tuple<int64_t, int64_t, double>>> per_k(nk);
    for (int64_t o = 0; o < out.size(); ++o) {
        DVec lo, hi;
        out.bounds(out.keys[o], lo, hi);
        const DVec s = out.size_at(out.keys[o].level);
        double Vo = 1.0;
        DVec c{};
        for (int d = 0; d < D; ++d) {
            Vo *= s[d];
            c[d] = lo[d] + 0.5 * s[d];
        }
        for (int64_t k = 0; k < nk; ++k) {
            DVec t{};
            Key cell = out.keys[o];
            int64_t rem = k;
            for (int d = D - 1; d >= 0; --d) {
                t[d] = c[d] + (static_cast<int>(rem % K) - r) * s[d];
                cell.coordinates[d] += static_cast<int>(rem % K) - r;
                rem /= K;
            }
            if (kc.mode == AdaptiveConvMode::OverlapWeighted) {
                DVec blo{}, bhi{};
                for (int d = 0; d < D; ++d) {
                    blo[d] = t[d] - 0.5 * s[d];
                    bhi[d] = t[d] + 0.5 * s[d];
                }
                in.for_each_overlap_scan(blo, bhi, [&](int64_t j, double ov) {
                    per_k[k].emplace_back(o, j, ov / Vo);
                });
            } else {
                const int64_t j = oracle_containing_center(in, idx, cell);
                const int64_t jf = oracle_containing(in, idx, t);
                if (j != jf && float_ties) {
                    ++*float_ties;
                    // A tie: the float answer is empty or a leaf touching t, and so is ours.
                    if ((jf >= 0 && !closed_contains(in, jf, t)) ||
                        (j >= 0 && !closed_contains(in, j, t)) || (j < 0 && jf < 0))
                        ++*bad_ties;
                }
                if (j < 0) continue;
                per_k[k].emplace_back(o, j,
                                      kc.mode == AdaptiveConvMode::VolumeWeighted
                                          ? in.volumes.data_ptr<double>()[j] / Vo
                                          : 1.0);
            }
        }
    }
    Pairs p;
    p.offsets.push_back(0);
    for (int64_t k = 0; k < nk; ++k) {
        for (auto& [o, j, w] : per_k[k]) {
            p.out.push_back(o);
            p.in.push_back(j);
            p.k.push_back(k);
            p.w.push_back(w);
        }
        p.offsets.push_back(static_cast<int64_t>(p.in.size()));
    }
    return p;
}

std::vector<int64_t> to_vec(const at::Tensor& t) {
    auto c = t.contiguous();
    return std::vector<int64_t>(c.data_ptr<int64_t>(), c.data_ptr<int64_t>() + c.numel());
}

}  // namespace

TEST(parallel_kmap_matches_serial_oracle) {
    std::mt19937_64 rng(424242);
    // Large enough that the map build runs on several threads (N * K^D >= 32768).
    struct Spec {
        int D;
        uint32_t depth;
        int extent;
        double p_refine, p_drop;
    };
    for (const Spec& s : {Spec{3, 3, 6, 0.25, 0.3}, Spec{2, 5, 20, 0.35, 0.2},
                          Spec{4, 2, 3, 0.2, 0.3}}) {
        auto topo = random_tree(s.D, s.depth, s.extent, s.p_refine, s.p_drop, rng);
        CHECK(topo->size() * 27 >= 32768 || s.D != 3);
        auto x = AdaptiveSparseTensor::from_topology(
            topo, torch::zeros({topo->size(), 1}, torch::kFloat64));
        for (auto mode : {AdaptiveConvMode::CenterSampled, AdaptiveConvMode::VolumeWeighted,
                          AdaptiveConvMode::OverlapWeighted}) {
            for (int shift : {0, -1, -2}) {
                AdaptiveKernelConfig kc;
                kc.mode = mode;
                kc.output_level_shift = shift;
                kc.kernel_size = s.D == 2 ? 5 : 3;
                KernelMapCache::instance().clear();
                auto e = KernelMapCache::instance().get_or_build(x, kc);
                // Output leaves equal the original set-based derivation.
                const auto expect_keys = shift == 0 ? topo->keys : oracle_output_keys(*topo, shift);
                CHECK(e.out_topology->keys == expect_keys);
                int64_t ties = 0, bad = 0;
                const Pairs p = oracle_pairs(*e.out_topology, *topo, kc, &ties, &bad);
                // Non-binary voxel sizes and origins: taps on corners shared by finer leaves
                // differ from the old float point location only by which touching leaf won.
                CHECK_EQ(bad, int64_t(0));
                std::cout << "    D=" << s.D << " mode=" << static_cast<int>(mode)
                          << " shift=" << shift << " pairs=" << p.in.size()
                          << " float_tie_changes=" << ties << std::endl;
                CHECK(e.map.kernel_offsets == p.offsets);
                CHECK(to_vec(e.map.in_index) == p.in);
                CHECK(to_vec(e.map.out_index) == p.out);
                CHECK(to_vec(e.map.kernel_index) == p.k);
                auto w = e.map.weight.contiguous();
                CHECK(std::vector<double>(w.data_ptr<double>(),
                                          w.data_ptr<double>() + w.numel()) == p.w);
                // A second build (fresh cache) is bit-identical.
                KernelMapCache::instance().clear();
                auto e2 = KernelMapCache::instance().get_or_build(x, kc);
                CHECK(torch::equal(e.map.in_index, e2.map.in_index));
                CHECK(torch::equal(e.map.weight, e2.map.weight));
            }
        }
    }
    KernelMapCache::instance().clear();
}

TEST(find_containing_matches_level_probe_oracle) {
    std::mt19937_64 rng(99);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (int D : {2, 3, 4}) {
        auto t = random_tree(D, 5, D == 4 ? 2 : 3, D == 4 ? 0.15 : 0.3, 0.3, rng);
        const auto idx = key_index(*t);
        int64_t found = 0;
        for (int n = 0; n < 20000; ++n) {
            DVec p{};
            for (int d = 0; d < D; ++d) p[d] = (u01(rng) - 0.5) * 8.0 * t->base[d];
            if (n % 97 == 0) p[n % D] = n % 2 ? nan : inf;
            if (n % 89 == 0) p[n % D] = 1e30;
            if (n % 5 == 0) {  // exactly on a leaf corner (grid line)
                DVec lo, hi;
                t->bounds(t->keys[n % t->size()], lo, hi);
                p = (n / 5) % 2 ? lo : hi;
            }
            const int64_t a = t->find_containing(p);
            const int64_t b = oracle_containing(*t, idx, p);
            if (a != b) FAIL_AT("point " << n << ": " << a << " vs oracle " << b);
            found += a >= 0;
        }
        CHECK(found > 1000);
        // Exact-key find agrees with the key index, including absent and internal keys.
        for (int64_t i = 0; i < t->size(); ++i) {
            CHECK_EQ(t->find(t->keys[i]), i);
            Key parent = adt::parent_key(t->keys[i]);
            if (t->keys[i].level > 0 && !idx.count([&] {
                    std::vector<int64_t> key{static_cast<int64_t>(parent.level)};
                    for (int d = 0; d < D; ++d) key.push_back(parent.coordinates[d]);
                    return key;
                }()))
                CHECK_EQ(t->find(parent), int64_t(-1));
        }
    }
}

TEST(parallel_interpolate_matches_serial_oracle) {
    std::mt19937_64 rng(5150);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    auto topo = random_tree(3, 4, 4, 0.3, 0.2, rng);
    auto feats = torch::randn({topo->size(), 3}, torch::kFloat64);
    auto x = AdaptiveSparseTensor::from_topology(topo, feats);
    const int64_t M = 60000;  // enough points for several worker threads
    auto pts = torch::empty({M, 3}, torch::kFloat64);
    double* pp = pts.data_ptr<double>();
    for (int64_t i = 0; i < M * 3; ++i) pp[i] = (u01(rng) - 0.5) * 9.0 * topo->base[i % 3];
    const auto idx = key_index(*topo);
    for (auto mode : {InterpMode::Nearest, InterpMode::Multilinear}) {
        // Serial oracle with the same arithmetic and accumulation order.
        std::vector<int64_t> pt, lf;
        std::vector<double> w;
        for (int64_t m = 0; m < M; ++m) {
            DVec q{};
            for (int d = 0; d < 3; ++d) q[d] = pp[m * 3 + d];
            const int64_t j = oracle_containing(*topo, idx, q);
            if (j < 0) continue;
            if (mode == InterpMode::Nearest) {
                pt.push_back(m);
                lf.push_back(j);
                w.push_back(1.0);
                continue;
            }
            const DVec s = topo->size_at(topo->keys[j].level);
            DVec lo{}, hi{};
            for (int d = 0; d < 3; ++d) {
                lo[d] = q[d] - 0.5 * s[d];
                hi[d] = q[d] + 0.5 * s[d];
            }
            std::vector<std::pair<int64_t, double>> hits;
            double total = 0.0;
            topo->for_each_overlap_scan(lo, hi, [&](int64_t leaf, double ov) {
                hits.emplace_back(leaf, ov);
                total += ov;
            });
            for (auto& h : hits) {
                pt.push_back(m);
                lf.push_back(h.first);
                w.push_back(h.second / total);
            }
        }
        CHECK(pt.size() > 1000);
        const int64_t n = static_cast<int64_t>(pt.size());
        auto expected = torch::zeros({M, 3}, torch::kFloat64);
        expected.index_add_(0, torch::from_blob(pt.data(), {n}, torch::kInt64).clone(),
                            feats.index_select(0, torch::from_blob(lf.data(), {n}, torch::kInt64).clone()) *
                                torch::from_blob(w.data(), {n}, torch::kFloat64).clone().unsqueeze(1));
        auto got = interpolate(x, pts, mode);
        CHECK(torch::equal(got, expected));
    }
}
