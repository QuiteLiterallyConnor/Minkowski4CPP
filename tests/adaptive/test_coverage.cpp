// Coverage tests: brute-force references for convolution in D = 2, 3, 4 and kernel size 5,
// finite-difference gradient checks for the remaining differentiable operations, empty
// tensors, and a topology large enough to take the parallel host / CUDA topology paths.
//
// Tolerances (all geometry is dyadic, so kernel maps are exact and only feature arithmetic
// rounds):
// - float64 forward vs brute force: rtol = atol = 1e-9 (different summation order only).
// - float64 gradients vs central differences, eps = 1e-6: rtol 1e-5, atol 1e-6. Every checked
//   map is linear (or piecewise linear away from Max ties), so the difference error is pure
//   rounding, about 1e-16 * |loss| / eps ~ 1e-10.
// - float32 CPU vs CUDA (large topology): forward and interpolation/pooling 1e-4 / 1e-5;
//   gradients rtol 1e-3 (feature grad atol 1e-3, weight grad atol 1e-2, since weight grads sum
//   tens of thousands of float32 products accumulated in different orders and with atomics).
#include <algorithm>
#include <array>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "MinkowskiEngine/adaptive/adaptive.hpp"
#include "utils/test_framework.hpp"

using namespace minkowski::adaptive;

namespace {

using KeyVec = std::vector<int64_t>;  // D coordinates followed by the level

AdaptiveVoxelConfig config_nd(std::vector<double> base, uint32_t depth,
                              std::vector<double> origin = {}) {
    AdaptiveVoxelConfig c;
    c.base_voxel_size = std::move(base);
    c.max_depth = depth;
    c.origin = std::move(origin);
    return c;
}

AdaptiveSparseTensor from_keys(const std::vector<KeyVec>& keys, const at::Tensor& features,
                               const AdaptiveVoxelConfig& cfg) {
    const int D = static_cast<int>(cfg.dimension());
    const int64_t n = static_cast<int64_t>(keys.size());
    auto c = torch::zeros({n, D}, torch::kInt64);
    auto l = torch::zeros({n}, torch::kInt32);
    auto ca = c.accessor<int64_t, 2>();
    auto la = l.accessor<int32_t, 1>();
    for (int64_t i = 0; i < n; ++i) {
        for (int d = 0; d < D; ++d) ca[i][d] = keys[i][d];
        la[i] = static_cast<int32_t>(keys[i][D]);
    }
    return AdaptiveSparseTensor::from_voxels(c, l, features, cfg);
}

// Random non-overlapping leaves below the level-0 cells [0, roots[d]) of every axis. Each cell
// becomes a leaf, stays empty (below level 0 only) or is split into its 2^D children.
std::vector<KeyVec> random_keys(int D, std::vector<int64_t> roots, uint32_t depth, uint32_t seed,
                                double p_leaf = 0.5, double p_empty = 0.15) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<KeyVec> keys;
    std::function<void(KeyVec, uint32_t)> visit = [&](KeyVec c, uint32_t level) {
        const double r = u(rng);
        if (level > 0 && r < p_empty) return;
        if (level == depth || r < p_empty + p_leaf) {
            c.push_back(level);
            keys.push_back(c);
            return;
        }
        for (int i = 0; i < (1 << D); ++i) {
            KeyVec k(D);
            for (int d = 0; d < D; ++d) k[d] = 2 * c[d] + ((i >> d) & 1);
            visit(k, level + 1);
        }
    };
    KeyVec cur(D, 0);
    std::function<void(int)> roots_rec = [&](int d) {
        if (d == D) {
            visit(cur, 0);
            return;
        }
        for (int64_t x = 0; x < roots[d]; ++x) {
            cur[d] = x;
            roots_rec(d + 1);
        }
    };
    roots_rec(0);
    return keys;
}

// Independent brute-force convolution: linear scans over all leaves, no hashing, no tree logic.
// Tap k of output leaf o is centre(o) + offset_k * size(o), first dimension slowest.
// `out_leaves` supplies the output leaves (the input leaves unless the convolution is strided).
at::Tensor reference_conv(const AdaptiveSparseTensor& t, const AdaptiveSparseTensor& out_leaves,
                          const at::Tensor& w, int K, AdaptiveConvMode mode) {
    const int D = t.dimension();
    const int64_t N = t.size();
    const int64_t NO = out_leaves.size();
    const int r = (K - 1) / 2;
    int64_t nk = 1;
    for (int d = 0; d < D; ++d) nk *= K;
    auto [lo_t, hi_t] = t.bounds();
    auto lo = lo_t.accessor<double, 2>();
    auto hi = hi_t.accessor<double, 2>();
    auto [olo_t, ohi_t] = out_leaves.bounds();
    auto olo = olo_t.accessor<double, 2>();
    auto ohi = ohi_t.accessor<double, 2>();
    auto x = t.features().to(torch::kFloat64);
    auto out = torch::zeros({NO, w.size(2)}, torch::kFloat64);
    for (int64_t k = 0; k < nk; ++k) {
        std::vector<int> off(D);
        int64_t rem = k;
        for (int d = D - 1; d >= 0; --d) {
            off[d] = static_cast<int>(rem % K) - r;
            rem /= K;
        }
        auto M = torch::zeros({NO, N}, torch::kFloat64);
        auto Ma = M.accessor<double, 2>();
        for (int64_t o = 0; o < NO; ++o) {
            double Vo = 1.0;
            std::vector<double> tap(D), s(D);
            for (int d = 0; d < D; ++d) {
                s[d] = ohi[o][d] - olo[o][d];
                Vo *= s[d];
                tap[d] = olo[o][d] + 0.5 * s[d] + off[d] * s[d];
            }
            for (int64_t j = 0; j < N; ++j) {
                double Vj = 1.0;
                for (int d = 0; d < D; ++d) Vj *= hi[j][d] - lo[j][d];
                if (mode == AdaptiveConvMode::OverlapWeighted) {
                    double ov = 1.0;
                    for (int d = 0; d < D; ++d)
                        ov *= std::max(0.0, std::min(hi[j][d], tap[d] + 0.5 * s[d]) -
                                                std::max(lo[j][d], tap[d] - 0.5 * s[d]));
                    if (ov > 0.0) Ma[o][j] += ov / Vo;
                } else {
                    bool inside = true;
                    for (int d = 0; d < D; ++d)
                        inside = inside && lo[j][d] <= tap[d] && tap[d] < hi[j][d];
                    if (inside)
                        Ma[o][j] += mode == AdaptiveConvMode::VolumeWeighted ? Vj / Vo : 1.0;
                }
            }
        }
        out += M.matmul(x).matmul(w[k].to(torch::kFloat64));
    }
    return out;
}

at::Tensor numerical_grad(const std::function<double(const at::Tensor&)>& loss, at::Tensor x,
                          double eps = 1e-6) {
    auto g = torch::zeros_like(x);
    auto flat = x.view(-1);
    auto gflat = g.view(-1);
    for (int64_t i = 0; i < flat.numel(); ++i) {
        double orig = flat[i].item<double>();
        flat[i] = orig + eps;
        double up = loss(x);
        flat[i] = orig - eps;
        double down = loss(x);
        flat[i] = orig;
        gflat[i] = (up - down) / (2 * eps);
    }
    return g;
}

// Autograd gradient of loss(x) w.r.t. x versus central finite differences.
void check_grad(const std::function<at::Tensor(const at::Tensor&)>& loss, const at::Tensor& x0,
                double rtol = 1e-5, double atol = 1e-6, const std::string& label = "") {
    auto x = x0.clone().set_requires_grad(true);
    auto l = loss(x);
    l.backward();
    CHECK(x.grad().defined());
    auto num = numerical_grad([&](const at::Tensor& v) { return loss(v).item<double>(); },
                              x0.clone());
    if (!torch::allclose(x.grad(), num, rtol, atol)) {
        auto d = (x.grad() - num).abs();
        auto i = d.argmax().item<int64_t>();
        std::cout << "         check_grad " << label << ": loss " << l.item<double>()
                  << ", max |grad| " << num.abs().max().item<double>() << ", worst element "
                  << i << " autograd " << x.grad().view(-1)[i].item<double>() << " vs fd "
                  << num.view(-1)[i].item<double>() << "\n";
    }
    CHECK_ALLCLOSE(x.grad(), num, rtol, atol);
}

const AdaptiveConvMode kModes[] = {AdaptiveConvMode::CenterSampled,
                                   AdaptiveConvMode::VolumeWeighted,
                                   AdaptiveConvMode::OverlapWeighted};

void check_conv_against_reference(int D, std::vector<double> base, std::vector<double> origin,
                                  std::vector<int64_t> roots, uint32_t depth, int K,
                                  uint32_t seed) {
    auto keys = random_keys(D, roots, depth, seed);
    torch::manual_seed(seed);
    auto t = from_keys(keys, torch::randn({int64_t(keys.size()), 2}, torch::kFloat64),
                       config_nd(base, depth, origin));
    int64_t nk = 1;
    for (int d = 0; d < D; ++d) nk *= K;
    auto w = torch::randn({nk, 2, 3}, torch::kFloat64);
    for (auto mode : kModes) {
        AdaptiveKernelConfig kc;
        kc.kernel_size = K;
        kc.mode = mode;
        auto out = adaptive_convolution(t, w, kc);
        CHECK_EQ(out.size(), t.size());
        CHECK_ALLCLOSE(out.features(), reference_conv(t, t, w, K, mode), 1e-9, 1e-9);
        // Strided: outputs one and two levels coarser, same tap rule on the output leaves.
        for (int shift : {-1, -2}) {
            kc.output_level_shift = shift;
            auto down = adaptive_convolution(t, w, kc);
            CHECK(!down.has_overlapping_leaves());
            CHECK_ALLCLOSE(down.features(), reference_conv(t, down, w, K, mode), 1e-9, 1e-9);
        }
    }
}

// Checkerboard of level-0 leaves and fully refined level-2 blocks: every coarse leaf borders
// fine leaves on all sides, so coarse taps land on fine-cell vertices and fine taps fall
// inside coarse leaves (the mixed-resolution boundary cases).
TEST(coverage_conv_level_boundaries_match_bruteforce) {
    std::vector<KeyVec> keys;
    for (int64_t x = 0; x < 4; ++x)
        for (int64_t y = 0; y < 4; ++y)
            for (int64_t z = 0; z < 2; ++z) {
                if ((x + y + z) % 2 == 0) {
                    keys.push_back({x, y, z, 0});
                    continue;
                }
                for (int64_t a = 0; a < 4; ++a)
                    for (int64_t b = 0; b < 4; ++b)
                        for (int64_t c = 0; c < 4; ++c)
                            keys.push_back({4 * x + a, 4 * y + b, 4 * z + c, 2});
            }
    torch::manual_seed(31);
    auto t = from_keys(keys, torch::randn({int64_t(keys.size()), 2}, torch::kFloat64),
                       config_nd({4.0, 4.0, 4.0}, 2, {-3.0, 1.0, 0.5}));
    auto w = torch::randn({27, 2, 2}, torch::kFloat64);
    for (auto mode : kModes)
        for (int shift : {0, -1}) {
            AdaptiveKernelConfig kc;
            kc.mode = mode;
            kc.output_level_shift = shift;
            auto out = adaptive_convolution(t, w, kc);
            CHECK_ALLCLOSE(out.features(), reference_conv(t, out, w, 3, mode), 1e-9, 1e-9);
        }
}

}  // namespace

// ---------------------------------------------------------------- convolution references

TEST(coverage_conv_3d_mixed_tree_matches_bruteforce) {
    check_conv_against_reference(3, {8.0, 8.0, 8.0}, {}, {3, 2, 2}, 3, 3, 101);
}

TEST(coverage_conv_2d_matches_bruteforce) {
    check_conv_against_reference(2, {2.0, 1.0}, {-0.5, 0.25}, {4, 3}, 3, 3, 202);
}

TEST(coverage_conv_4d_matches_bruteforce) {
    check_conv_against_reference(4, {1.0, 1.0, 2.0, 1.0}, {}, {2, 2, 1, 1}, 2, 3, 303);
}

TEST(coverage_conv_kernel_size_5_matches_bruteforce) {
    check_conv_against_reference(3, {8.0, 8.0, 8.0}, {}, {3, 2, 2}, 2, 5, 404);
    check_conv_against_reference(2, {1.0, 1.0}, {}, {4, 4}, 3, 5, 405);
}

TEST(coverage_conv_kernel_size_1_is_pointwise) {
    auto keys = random_keys(3, {2, 2, 2}, 2, 7);
    torch::manual_seed(7);
    auto t = from_keys(keys, torch::randn({int64_t(keys.size()), 3}, torch::kFloat64),
                       config_nd({4.0, 4.0, 4.0}, 2));
    auto w = torch::randn({1, 3, 2}, torch::kFloat64);
    AdaptiveKernelConfig kc;
    kc.kernel_size = 1;
    CHECK_ALLCLOSE(adaptive_convolution(t, w, kc).features(), t.features().matmul(w[0]), 1e-12,
                   1e-12);
}

TEST(coverage_conv_rejects_even_kernel_and_bad_weight) {
    auto t = from_keys({{0, 0, 0, 0}}, torch::ones({1, 2}, torch::kFloat64),
                       config_nd({1.0, 1.0, 1.0}, 1));
    AdaptiveKernelConfig kc;
    kc.kernel_size = 4;
    CHECK_THROWS(adaptive_convolution(t, torch::ones({64, 2, 1}, torch::kFloat64), kc));
    kc.kernel_size = 3;
    CHECK_THROWS(adaptive_convolution(t, torch::ones({27, 3, 1}, torch::kFloat64), kc));
    CHECK_THROWS(adaptive_convolution(t, torch::ones({9, 2, 1}, torch::kFloat64), kc));
}

// ---------------------------------------------------------------- gradients

namespace {

AdaptiveSparseTensor gradient_tree(int64_t channels = 2) {
    auto keys = random_keys(3, {2, 2, 1}, 2, 55);
    torch::manual_seed(55);
    return from_keys(keys, torch::randn({int64_t(keys.size()), channels}, torch::kFloat64),
                     config_nd({8.0, 8.0, 8.0}, 2));
}

}  // namespace

TEST(coverage_gradient_refine_split_matches_finite_differences) {
    auto base = gradient_tree();
    auto region = AxisAlignedRegion{{0, 0, 0}, {12, 12, 8}};
    auto probe_tensor = base;
    probe_tensor.refine(region, 2, RefinePolicy::Split);
    torch::manual_seed(1);
    auto probe = torch::randn({probe_tensor.size(), 2}, torch::kFloat64);
    check_grad(
        [&](const at::Tensor& f) {
            auto t = base.with_features(f);
            t.refine(region, 2, RefinePolicy::Split);
            return (t.features() * probe).sum();
        },
        base.features());
}

TEST(coverage_gradient_refine_replicate_matches_finite_differences) {
    auto base = gradient_tree();
    auto region = AxisAlignedRegion{{4, 4, 0}, {12, 12, 8}};
    auto shape = base;
    shape.refine(region, 2, RefinePolicy::Replicate);
    torch::manual_seed(2);
    auto probe = torch::randn({shape.size(), 2}, torch::kFloat64);
    check_grad(
        [&](const at::Tensor& f) {
            auto t = base.with_features(f);
            t.refine(region, 2, RefinePolicy::Replicate);
            return (t.features() * probe).sum();
        },
        base.features());
}

TEST(coverage_gradient_strided_and_transposed_convolution) {
    auto base = gradient_tree();
    for (int shift : {0, -1, -2}) {
        for (auto mode : kModes) {
            AdaptiveKernelConfig kc;
            kc.mode = mode;
            kc.output_level_shift = shift;
            const std::string lbl = "shift " + std::to_string(shift) + " mode " + std::to_string(int(mode));
            torch::manual_seed(10 - shift);
            auto w = torch::randn({27, 2, 3}, torch::kFloat64);
            auto down = adaptive_convolution(base, w, kc);
            auto probe_down = torch::randn({down.size(), 3}, torch::kFloat64);
            // Strided forward, w.r.t. features and weight.
            check_grad(
                [&](const at::Tensor& f) {
                    return (adaptive_convolution(base.with_features(f), w, kc).features() *
                            probe_down)
                        .sum();
                },
                base.features(), 1e-5, 1e-6, lbl + " forward/features");
            check_grad(
                [&](const at::Tensor& v) {
                    return (adaptive_convolution(base, v, kc).features() * probe_down).sum();
                },
                w, 1e-5, 1e-6, lbl + " forward/weight");
            // Transpose back onto the reference leaves, w.r.t. features and weight.
            auto wt = torch::randn({27, 3, 2}, torch::kFloat64);
            auto probe_up = torch::randn({base.size(), 2}, torch::kFloat64);
            check_grad(
                [&](const at::Tensor& f) {
                    return (adaptive_convolution_transpose(down.with_features(f), wt, kc, base)
                                .features() *
                            probe_up)
                        .sum();
                },
                down.features(), 1e-5, 1e-6, lbl + " transpose/features");
            check_grad(
                [&](const at::Tensor& v) {
                    return (adaptive_convolution_transpose(down, v, kc, base).features() *
                            probe_up)
                        .sum();
                },
                wt, 1e-5, 1e-6, lbl + " transpose/weight");
        }
    }
}

TEST(coverage_transposed_convolution_is_the_adjoint_of_the_forward_map) {
    // <conv(x; W), y> == <x, conv_T(y; W^T)> with W^T[k] = W[k]^T.
    auto base = gradient_tree();
    for (int shift : {0, -1}) {
        for (auto mode : kModes) {
            AdaptiveKernelConfig kc;
            kc.mode = mode;
            kc.output_level_shift = shift;
            torch::manual_seed(20);
            auto w = torch::randn({27, 2, 3}, torch::kFloat64);
            auto y_shape = adaptive_convolution(base, w, kc);
            auto y = torch::randn({y_shape.size(), 3}, torch::kFloat64);
            const double lhs = (y_shape.features() * y).sum().item<double>();
            auto back = adaptive_convolution_transpose(y_shape.with_features(y),
                                                       w.transpose(1, 2).contiguous(), kc, base);
            const double rhs = (base.features() * back.features()).sum().item<double>();
            CHECK_NEAR(lhs, rhs, 1e-9 * (1.0 + std::fabs(lhs)));
        }
    }
}

TEST(coverage_gradient_interpolation_nearest) {
    auto base = gradient_tree();
    torch::manual_seed(3);
    // Interior points (never on a leaf face), some outside every leaf.
    auto pts = torch::rand({40, 3}, torch::kFloat64) * 20.0 - 2.0;
    pts = torch::floor(pts) + 0.37;
    auto probe = torch::randn({40, 2}, torch::kFloat64);
    check_grad(
        [&](const at::Tensor& f) {
            return (interpolate(base.with_features(f), pts, InterpMode::Nearest) * probe).sum();
        },
        base.features());
}

TEST(coverage_gradient_interpolation_multilinear_mixed_tree) {
    auto base = gradient_tree();
    torch::manual_seed(4);
    auto pts = torch::floor(torch::rand({40, 3}, torch::kFloat64) * 20.0 - 2.0) + 0.37;
    auto probe = torch::randn({40, 2}, torch::kFloat64);
    check_grad(
        [&](const at::Tensor& f) {
            return (interpolate(base.with_features(f), pts, InterpMode::Multilinear) * probe)
                .sum();
        },
        base.features());
}

// Multilinear averages over a box the size of the containing leaf. Across a face between
// equal-level leaves that box moves continuously; across a level change its size jumps, and so
// can the value. Coarse [0,2)^2 = 0, fine [2,3)x[0,1) = 0, fine [2,3)x[1,2) = 1:
// just left of x = 2 the 2x2 box sees 0.5 area of value 1 out of 3 -> 1/6; just right the 1x1
// box sees none -> 0.
TEST(coverage_interpolation_multilinear_continuity_by_level) {
    auto t = from_keys({{0, 0, 0}, {2, 0, 1}, {2, 1, 1}},
                       torch::tensor({{0.0}, {0.0}, {1.0}}, torch::kFloat64),
                       config_nd({2.0, 2.0}, 1));
    const double e = 1e-7;
    auto at = [&](double x, double y) {
        return interpolate(t, torch::tensor({{x, y}}, torch::kFloat64), InterpMode::Multilinear)[0][0]
            .item<double>();
    };
    CHECK_NEAR(at(2.5, 1.0 - e), at(2.5, 1.0 + e), 1e-6);  // same level: continuous
    CHECK_NEAR(at(2.5, 1.0), 0.5, 1e-12);
    CHECK_NEAR(at(2.0 - e, 0.5), 1.0 / 6.0, 1e-6);  // level change: jumps from 1/6 ...
    CHECK_NEAR(at(2.0 + e, 0.5), 0.0, 1e-6);        // ... to 0
}

TEST(coverage_gradient_from_points_mean) {
    torch::manual_seed(5);
    auto pts = torch::rand({30, 3}, torch::kFloat64) * 16.0;
    auto f0 = torch::randn({30, 2}, torch::kFloat64);
    auto cfg = config_nd({4.0, 4.0, 4.0}, 2);
    auto shape = AdaptiveSparseTensor::from_points(pts, f0, cfg);
    CHECK(shape.size() < 30);  // some cells hold several points, so the mean is exercised
    auto probe = torch::randn({shape.size(), 2}, torch::kFloat64);
    check_grad(
        [&](const at::Tensor& f) {
            return (AdaptiveSparseTensor::from_points(pts, f, cfg).features() * probe).sum();
        },
        f0);
}

TEST(coverage_refine_by_variance_with_differentiable_point_features) {
    torch::manual_seed(7);
    auto pts = torch::rand({40, 3}, torch::kFloat64) * 16.0;
    auto f = torch::randn({40, 2}, torch::kFloat64).set_requires_grad(true);
    auto t = AdaptiveSparseTensor::from_points(pts, f, config_nd({8.0, 8.0, 8.0}, 2));
    t.refine_by_variance(0.1);
    CHECK(t.size() > 0);
    (t.features().sum()).backward();  // the graph still reaches f through Replicate refinement
    CHECK(f.grad().defined());
}

TEST(coverage_gradient_max_with_ties_splits_evenly) {
    // Parent (0,0,0)@0 with three occupied children: two tie at 5, one is 1.
    auto cfg = config_nd({2.0, 2.0, 2.0}, 1);
    auto f0 = torch::tensor({{5.0}, {1.0}, {5.0}}, torch::kFloat64);
    auto t = from_keys({{0, 0, 0, 1}, {1, 0, 0, 1}, {0, 1, 0, 1}}, f0, cfg);
    auto f = t.features().clone().set_requires_grad(true);
    auto pooled = adaptive_pool(t.with_features(f), 0, Reduction::Max);
    CHECK_NEAR(pooled.features()[0][0].item<double>(), 5.0, 0.0);
    pooled.features().sum().backward();
    auto g = f.grad();
    const int64_t a = t.index_of({0, 0, 0}, 1), b = t.index_of({1, 0, 0}, 1),
                  c = t.index_of({0, 1, 0}, 1);
    CHECK_NEAR(g[b][0].item<double>(), 0.0, 0.0);
    // A valid subgradient: the tied entries share the unit gradient.
    CHECK_NEAR(g[a][0].item<double>() + g[c][0].item<double>(), 1.0, 1e-12);
    CHECK_NEAR(g[a][0].item<double>(), g[c][0].item<double>(), 1e-12);
    // Moving all tied entries together is differentiable: directional FD equals sum of grads.
    const double eps = 1e-6;
    auto dir = torch::zeros_like(f0);
    dir[a][0] = 1.0;
    dir[c][0] = 1.0;
    auto at = [&](double e) {
        return adaptive_pool(t.with_features(t.features() + e * dir), 0, Reduction::Max)
            .features()[0][0]
            .item<double>();
    };
    CHECK_NEAR((at(eps) - at(-eps)) / (2 * eps), (g * dir).sum().item<double>(), 1e-6);
    // Same contract for resample_to and for a convolution followed by Max pooling.
    auto ref = from_keys({{0, 0, 0, 0}}, torch::zeros({1, 1}, torch::kFloat64), cfg);
    auto f2 = t.features().clone().set_requires_grad(true);
    t.with_features(f2).resample_to(ref, Reduction::Max).features().sum().backward();
    CHECK_NEAR(f2.grad().sum().item<double>(), 1.0, 1e-12);
    CHECK_NEAR(f2.grad()[b][0].item<double>(), 0.0, 0.0);
}

TEST(coverage_gradient_conv_2d_and_4d) {
    for (int D : {2, 4}) {
        std::vector<double> base(D, 2.0);
        std::vector<int64_t> roots(D, 2);
        auto keys = random_keys(D, roots, D == 2 ? 3 : 1, 77 + D);
        torch::manual_seed(D);
        auto t = from_keys(keys, torch::randn({int64_t(keys.size()), 2}, torch::kFloat64),
                           config_nd(base, D == 2 ? 3 : 1));
        int64_t nk = D == 2 ? 9 : 81;
        auto w = torch::randn({nk, 2, 2}, torch::kFloat64);
        auto probe = torch::randn({t.size(), 2}, torch::kFloat64);
        for (auto mode : kModes) {
            AdaptiveKernelConfig kc;
            kc.mode = mode;
            check_grad(
                [&](const at::Tensor& f) {
                    return (adaptive_convolution(t.with_features(f), w, kc).features() * probe)
                        .sum();
                },
                t.features());
            check_grad(
                [&](const at::Tensor& v) {
                    return (adaptive_convolution(t, v, kc).features() * probe).sum();
                },
                w);
        }
    }
}

// ---------------------------------------------------------------- empty tensors

TEST(coverage_empty_tensor_conv_pool_interp) {
    auto cfg = config_nd({8.0, 8.0, 8.0}, 2);
    auto t = AdaptiveSparseTensor::from_points(torch::zeros({0, 3}, torch::kFloat64),
                                               torch::zeros({0, 4}, torch::kFloat64), cfg);
    CHECK_EQ(t.size(), int64_t(0));
    auto w = torch::randn({27, 4, 5}, torch::kFloat64);
    for (auto mode : kModes) {
        for (int shift : {0, -1}) {
            AdaptiveKernelConfig kc;
            kc.mode = mode;
            kc.output_level_shift = shift;
            auto out = adaptive_convolution(t, w, kc);
            CHECK_EQ(out.size(), int64_t(0));
            CHECK_EQ(out.features().size(1), int64_t(5));
            auto back = adaptive_convolution_transpose(out, w.transpose(1, 2), kc, t);
            CHECK_EQ(back.size(), int64_t(0));
            CHECK_EQ(back.features().size(1), int64_t(4));
        }
    }
    for (auto r : {Reduction::Mean, Reduction::Sum, Reduction::Max}) {
        auto p = adaptive_pool(t, 0, r);
        CHECK_EQ(p.size(), int64_t(0));
        CHECK_EQ(p.features().size(1), int64_t(4));
    }
    auto g = adaptive_global_avg_pool(t);
    CHECK_EQ(g.numel(), int64_t(4));
    CHECK_NEAR(g.abs().sum().item<double>(), 0.0, 0.0);
    auto pts = torch::rand({3, 3}, torch::kFloat64);
    for (auto mode : {InterpMode::Nearest, InterpMode::Multilinear}) {
        auto v = interpolate(t, pts, mode);
        CHECK_EQ(v.size(0), int64_t(3));
        CHECK_EQ(v.size(1), int64_t(4));
        CHECK_NEAR(v.abs().sum().item<double>(), 0.0, 0.0);
        CHECK_EQ(interpolate(t, torch::zeros({0, 3}, torch::kFloat64), mode).size(0), int64_t(0));
    }
    auto u = t.resample_uniform(1);
    CHECK_EQ(u.size(), int64_t(0));
    CHECK_EQ(t.resample_to(t).size(), int64_t(0));
}

// ---------------------------------------------------------------- large topologies

namespace {

// ~40K leaves: a 12^3 block of level-0 cells, each kept at level 0, 1 or 2 (all children).
std::vector<KeyVec> large_keys(uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pick(0, 9);
    std::vector<KeyVec> keys;
    for (int64_t x = 0; x < 12; ++x)
        for (int64_t y = 0; y < 12; ++y)
            for (int64_t z = 0; z < 12; ++z) {
                const int v = pick(rng);
                const int level = v < 2 ? 0 : (v < 5 ? 1 : 2);
                const int64_t n = int64_t(1) << level;
                for (int64_t a = 0; a < n; ++a)
                    for (int64_t b = 0; b < n; ++b)
                        for (int64_t c = 0; c < n; ++c)
                            keys.push_back({x * n + a, y * n + b, z * n + c, level});
            }
    return keys;
}

bool key_less_vec(const KeyVec& a, const KeyVec& b) {
    if (a[3] != b[3]) return a[3] < b[3];
    return std::lexicographical_compare(a.begin(), a.begin() + 3, b.begin(), b.begin() + 3);
}

}  // namespace

// >16,384 keys takes the CUDA sort / ancestor check in CUDA builds with a GPU, and >32,768 keys
// takes the parallel host sort and ancestor check otherwise. Both must match a plain std::sort.
TEST(coverage_large_topology_sort_and_overlap_check) {
    auto keys = large_keys(11);
    CHECK(keys.size() > 32768);
    std::mt19937 rng(12);
    std::shuffle(keys.begin(), keys.end(), rng);
    const int64_t n = static_cast<int64_t>(keys.size());
    auto f = torch::arange(n, torch::kFloat64).unsqueeze(1);
    auto cfg = config_nd({8.0, 8.0, 8.0}, 2);
    auto t = from_keys(keys, f, cfg);
    CHECK_EQ(t.size(), n);

    std::vector<int64_t> order(n);
    for (int64_t i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](int64_t a, int64_t b) { return key_less_vec(keys[a], keys[b]); });
    auto coords = t.coordinates();
    auto levels = t.levels();
    auto ca = coords.accessor<int64_t, 2>();
    auto la = levels.accessor<int32_t, 1>();
    auto fa = t.features().accessor<double, 2>();
    bool ok = true;
    for (int64_t i = 0; i < n && ok; ++i) {
        const KeyVec& k = keys[order[i]];
        ok = ca[i][0] == k[0] && ca[i][1] == k[1] && ca[i][2] == k[2] && la[i] == k[3] &&
             fa[i][0] == double(order[i]);
    }
    CHECK(ok);
    CHECK(!t.has_overlapping_leaves());

    // Adding the level-0 ancestor of a refined cell, or a level-1 ancestor of a level-2 leaf,
    // must be rejected.
    for (int which : {0, 1}) {
        auto bad = keys;
        for (const auto& k : keys)
            if (k[3] == 2) {
                const int64_t s = which == 0 ? 2 : 1;
                bad.push_back({k[0] >> s, k[1] >> s, k[2] >> s, 2 - s});
                break;
            }
        CHECK_THROWS(from_keys(bad, torch::zeros({int64_t(bad.size()), 1}, torch::kFloat64), cfg));
    }
    // Duplicates are rejected as well.
    auto dup = keys;
    dup.push_back(keys[n / 2]);
    CHECK_THROWS(from_keys(dup, torch::zeros({int64_t(dup.size()), 1}, torch::kFloat64), cfg));
}

TEST(coverage_large_topology_refine_and_coarsen_roundtrip) {
    auto keys = large_keys(13);
    auto cfg = config_nd({8.0, 8.0, 8.0}, 2);
    torch::manual_seed(13);
    auto t = from_keys(keys, torch::randn({int64_t(keys.size()), 2}, torch::kFloat64), cfg);
    auto all = AxisAlignedRegion::everything(3);
    auto u = t;
    u.refine(all, 2, RefinePolicy::Replicate);
    CHECK_EQ(u.size(), int64_t(12 * 12 * 12 * 64));
    CHECK(!u.has_overlapping_leaves());
    u.coarsen(all, 0, Reduction::Mean);
    CHECK_EQ(u.size(), int64_t(12 * 12 * 12));
    auto direct = adaptive_pool(t, 0, Reduction::Mean);
    CHECK(torch::equal(u.coordinates(), direct.coordinates()));
    CHECK_ALLCLOSE(u.features(), direct.features(), 1e-12, 1e-12);
}

TEST(coverage_large_topology_cpu_cuda_agreement) {
    if (!torch::cuda::is_available()) throw mink_test::Skip("CUDA not available");
    auto keys = large_keys(17);
    auto cfg = config_nd({8.0, 8.0, 8.0}, 2);
    torch::manual_seed(17);
    auto t = from_keys(keys, torch::randn({int64_t(keys.size()), 4}, torch::kFloat32), cfg);
    auto w = torch::randn({27, 4, 8}, torch::kFloat32) * 0.1;
    auto pts = torch::rand({4096, 3}, torch::kFloat64) * 96.0;
    for (auto mode : kModes) {
        for (int shift : {0, -1}) {
            AdaptiveKernelConfig kc;
            kc.mode = mode;
            kc.output_level_shift = shift;
            auto cpu = adaptive_convolution(t, w, kc);
            auto fg = t.features().to(torch::kCUDA).set_requires_grad(true);
            auto wg = w.to(torch::kCUDA).set_requires_grad(true);
            auto gpu = adaptive_convolution(t.with_features(fg), wg, kc);
            CHECK(torch::equal(cpu.coordinates(), gpu.coordinates()));
            CHECK_ALLCLOSE(cpu.features(), gpu.features().cpu(), 1e-4, 1e-4);

            auto fc = t.features().clone().set_requires_grad(true);
            auto wc = w.clone().set_requires_grad(true);
            adaptive_convolution(t.with_features(fc), wc, kc).features().square().sum().backward();
            gpu.features().square().sum().backward();
            CHECK_ALLCLOSE(fc.grad(), fg.grad().cpu(), 1e-3, 1e-3);
            CHECK_ALLCLOSE(wc.grad(), wg.grad().cpu(), 1e-3, 1e-2);
        }
    }
    for (auto mode : {InterpMode::Nearest, InterpMode::Multilinear})
        CHECK_ALLCLOSE(interpolate(t, pts, mode),
                       interpolate(t.to(torch::kCUDA), pts, mode).cpu(), 1e-5, 1e-5);
    for (auto r : {Reduction::Mean, Reduction::Sum, Reduction::Max})
        CHECK_ALLCLOSE(adaptive_pool(t, 0, r).features(),
                       adaptive_pool(t.to(torch::kCUDA), 0, r).features().cpu(), 1e-5, 1e-5);
}
