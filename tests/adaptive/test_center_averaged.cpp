// AdaptiveConvMode::CenterAveraged (opt-in): coarse outputs over finer inputs average the
// finer leaves inside each tap cell instead of reading the single child whose corner is the
// tap point. Checked: the documented weights, gradients against finite differences, full
// gradient coverage for output_level_shift = -1, and equality with CenterSampled where no tap
// cell is split (uniform grids, fine outputs over coarse inputs).
#include <functional>
#include <map>
#include <utility>

#include "MinkowskiEngine/adaptive/adaptive.hpp"
#include "MinkowskiEngine/adaptive/adaptive_topology.hpp"
#include "utils/test_framework.hpp"

using namespace minkowski::adaptive;

namespace {

AdaptiveVoxelConfig ca_config(uint32_t depth) {
    AdaptiveVoxelConfig c;
    c.base_voxel_size = {8.0, 8.0, 8.0};
    c.max_depth = depth;
    return c;
}

// Coarse level-0 leaves next to a fully refined (level 1) and a partly refined (level 2) block.
AdaptiveSparseTensor ca_mixed(int64_t channels) {
    std::vector<std::array<int64_t, 4>> v = {{0, 0, 0, 0}, {0, 1, 0, 0}, {1, 1, 0, 0}};
    for (int x = 2; x <= 3; ++x)
        for (int y = 0; y <= 1; ++y)
            for (int z = 0; z <= 1; ++z) v.push_back({x, y, z, 1});
    // Under level-0 cell (2, 0, 0): level-2 leaves below level-1 cell (4, 0, 0) (not all
    // present) and two of its level-1 siblings.
    for (int x = 8; x <= 9; ++x)
        for (int y = 0; y <= 1; ++y)
            for (int z = 0; z <= 1; ++z)
                if ((x + y + z) % 3 != 0) v.push_back({x, y, z, 2});
    v.push_back({5, 0, 0, 1});
    v.push_back({4, 1, 1, 1});
    const int64_t n = static_cast<int64_t>(v.size());
    auto c = torch::zeros({n, 3}, torch::kInt64);
    auto l = torch::zeros({n}, torch::kInt32);
    for (int64_t i = 0; i < n; ++i) {
        for (int d = 0; d < 3; ++d) c[i][d] = v[i][d];
        l[i] = static_cast<int>(v[i][3]);
    }
    torch::manual_seed(17);
    return AdaptiveSparseTensor::from_voxels(c, l, torch::randn({n, channels}, torch::kFloat64),
                                             ca_config(2));
}

at::Tensor ca_numerical_grad(const std::function<double(const at::Tensor&)>& loss, at::Tensor x,
                             double eps = 1e-6) {
    auto g = torch::zeros_like(x);
    auto flat = x.view(-1);
    auto gflat = g.view(-1);
    for (int64_t i = 0; i < flat.numel(); ++i) {
        const double orig = flat[i].item<double>();
        flat[i] = orig + eps;
        const double up = loss(x);
        flat[i] = orig - eps;
        const double down = loss(x);
        flat[i] = orig;
        gflat[i] = (up - down) / (2 * eps);
    }
    return g;
}

AdaptiveKernelConfig ca_kc(AdaptiveConvMode mode, int shift, int K = 3) {
    AdaptiveKernelConfig kc;
    kc.mode = mode;
    kc.output_level_shift = shift;
    kc.kernel_size = K;
    return kc;
}

}  // namespace

TEST(center_averaged_weights_partition_each_split_tap) {
    auto t = ca_mixed(1);
    for (int shift : {0, -1, -2}) {
        auto kc = ca_kc(AdaptiveConvMode::CenterAveraged, shift);
        auto m = build_adaptive_kernel_map(t, kc);
        auto sampled = build_adaptive_kernel_map(t, ca_kc(AdaptiveConvMode::CenterSampled, shift));
        auto in = m.in_index.contiguous();
        auto out = m.out_index.contiguous();
        auto ki = m.kernel_index.contiguous();
        auto w = m.weight.contiguous();
        std::map<std::pair<int64_t, int64_t>, double> sum;  // (slot, out) -> weight sum
        std::map<std::pair<int64_t, int64_t>, int64_t> count;
        for (int64_t p = 0; p < in.numel(); ++p) {
            CHECK(w.data_ptr<double>()[p] > 0.0);
            sum[{ki.data_ptr<int64_t>()[p], out.data_ptr<int64_t>()[p]}] += w.data_ptr<double>()[p];
            ++count[{ki.data_ptr<int64_t>()[p], out.data_ptr<int64_t>()[p]}];
        }
        // Every (slot, out) tap is a weighted average: weights sum to one.
        for (auto& [key, s] : sum) CHECK_NEAR(s, 1.0, 1e-12);
        // Single-leaf taps carry weight exactly 1 (the CenterSampled pair).
        for (int64_t p = 0; p < in.numel(); ++p)
            if (count[{ki.data_ptr<int64_t>()[p], out.data_ptr<int64_t>()[p]}] == 1)
                CHECK_EQ(w.data_ptr<double>()[p], 1.0);
        // Every tap CenterSampled reads is also read here (possibly together with siblings).
        auto sk = sampled.kernel_index.contiguous();
        auto so = sampled.out_index.contiguous();
        for (int64_t p = 0; p < sk.numel(); ++p)
            CHECK(sum.count({sk.data_ptr<int64_t>()[p], so.data_ptr<int64_t>()[p]}) == 1);
        CHECK(in.numel() >= sk.numel());
    }
}

TEST(center_averaged_shift_minus_one_reaches_every_input_leaf) {
    auto t = ca_mixed(2);
    for (int K : {1, 3}) {
        for (auto mode : {AdaptiveConvMode::CenterSampled, AdaptiveConvMode::CenterAveraged}) {
            auto f = t.features().clone().set_requires_grad(true);
            auto w = torch::randn({K * K * K, 2, 3}, torch::kFloat64);
            auto y = adaptive_convolution(t.with_features(f), w, ca_kc(mode, -1, K));
            y.features().pow(2).sum().backward();
            const int64_t zero_rows =
                (f.grad().abs().sum(1) == 0).sum().item<int64_t>();
            if (mode == AdaptiveConvMode::CenterAveraged) {
                CHECK_EQ(zero_rows, int64_t(0));
            } else if (K == 1) {
                // The default reads only the child at the tap corner, so 7 of the 8 level-1
                // leaves under (1, 0, 0) get no gradient. Existing behaviour, kept unchanged.
                CHECK(zero_rows > 0);
            }
        }
    }
}

TEST(center_averaged_gradients_match_finite_differences) {
    auto base = ca_mixed(2);
    for (int shift : {0, -1}) {
        auto kc = ca_kc(AdaptiveConvMode::CenterAveraged, shift);
        torch::manual_seed(23 + shift);
        auto weight0 = torch::randn({27, 2, 3}, torch::kFloat64);
        auto feats0 = base.features().clone();
        const int64_t n_out = adaptive_convolution(base, weight0, kc).size();
        auto probe = torch::randn({n_out, 3}, torch::kFloat64);
        auto loss_of = [&](const at::Tensor& f, const at::Tensor& w) {
            return (adaptive_convolution(base.with_features(f), w, kc).features() * probe).sum();
        };
        auto f = feats0.clone().set_requires_grad(true);
        auto w = weight0.clone().set_requires_grad(true);
        loss_of(f, w).backward();
        auto num_f = ca_numerical_grad(
            [&](const at::Tensor& x) { return loss_of(x, weight0).item<double>(); }, feats0.clone());
        auto num_w = ca_numerical_grad(
            [&](const at::Tensor& x) { return loss_of(feats0, x).item<double>(); },
            weight0.clone());
        CHECK_ALLCLOSE(f.grad(), num_f, 1e-5, 1e-6);
        CHECK_ALLCLOSE(w.grad(), num_w, 1e-5, 1e-6);

        // Transposed convolution back onto the input leaves.
        auto coarse = adaptive_convolution(base, weight0, kc);
        auto wt0 = torch::randn({27, 3, 2}, torch::kFloat64);
        auto probe_t = torch::randn({base.size(), 2}, torch::kFloat64);
        auto loss_t = [&](const at::Tensor& f2) {
            return (adaptive_convolution_transpose(coarse.with_features(f2), wt0, kc, base)
                        .features() *
                    probe_t)
                .sum();
        };
        auto g = coarse.features().clone().set_requires_grad(true);
        loss_t(g).backward();
        auto num_g = ca_numerical_grad([&](const at::Tensor& x) { return loss_t(x).item<double>(); },
                                       coarse.features().clone());
        CHECK_ALLCLOSE(g.grad(), num_g, 1e-5, 1e-6);
    }
}

TEST(center_averaged_equals_center_sampled_without_split_taps) {
    // Uniform level-1 grid with holes: every tap cell is a leaf or empty.
    std::vector<std::array<int64_t, 3>> v;
    for (int x = 0; x < 5; ++x)
        for (int y = 0; y < 4; ++y)
            for (int z = 0; z < 3; ++z)
                if ((x * 7 + y * 3 + z) % 4 != 0) v.push_back({x, y, z});
    const int64_t n = static_cast<int64_t>(v.size());
    auto c = torch::zeros({n, 3}, torch::kInt64);
    for (int64_t i = 0; i < n; ++i)
        for (int d = 0; d < 3; ++d) c[i][d] = v[i][d];
    auto t = AdaptiveSparseTensor::from_voxels(c, torch::ones({n}, torch::kInt32),
                                               torch::randn({n, 2}, torch::kFloat64),
                                               ca_config(2));
    for (int K : {3, 5}) {
        auto a = build_adaptive_kernel_map(t, ca_kc(AdaptiveConvMode::CenterAveraged, 0, K));
        auto b = build_adaptive_kernel_map(t, ca_kc(AdaptiveConvMode::CenterSampled, 0, K));
        CHECK(torch::equal(a.in_index, b.in_index));
        CHECK(torch::equal(a.out_index, b.out_index));
        CHECK(torch::equal(a.weight, b.weight));
        auto w = torch::randn({K * K * K, 2, 4}, torch::kFloat64);
        CHECK_ALLCLOSE(
            adaptive_convolution(t, w, ca_kc(AdaptiveConvMode::CenterAveraged, 0, K)).features(),
            adaptive_convolution(t, w, ca_kc(AdaptiveConvMode::CenterSampled, 0, K)).features(),
            1e-12, 1e-12);
    }
    // Fine outputs over coarse inputs: the containing leaf is coarser, so nothing is averaged.
    auto mixed = ca_mixed(1);
    auto m = build_adaptive_kernel_map(mixed, ca_kc(AdaptiveConvMode::CenterAveraged, 0));
    auto s = build_adaptive_kernel_map(mixed, ca_kc(AdaptiveConvMode::CenterSampled, 0));
    const auto& topo = *mixed.topology();
    auto mo = m.out_index.contiguous();
    auto mw = m.weight.contiguous();
    for (int64_t p = 0; p < mo.numel(); ++p)
        if (topo.keys[mo.data_ptr<int64_t>()[p]].level == 2)
            CHECK_EQ(mw.data_ptr<double>()[p], 1.0);
    CHECK(m.in_index.numel() > s.in_index.numel());  // the coarse leaves do average
}
