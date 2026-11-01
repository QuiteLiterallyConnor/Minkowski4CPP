// apply_map batching + per-device kernel-map cache: outputs and gradients must equal the
// original one-scatter-per-kernel-slot formulation, and repeated calls must reuse the cached
// device arrays.
#include <vector>

#include "MinkowskiEngine/adaptive/adaptive.hpp"
#include "MinkowskiEngine/adaptive/adaptive_topology.hpp"
#include "utils/test_framework.hpp"

using namespace minkowski::adaptive;

namespace {

AdaptiveSparseTensor mixed_tensor(int64_t channels, torch::ScalarType dtype) {
    torch::manual_seed(3);
    auto pts = torch::cat({torch::rand({600, 3}, torch::kFloat64) * 32.0,
                           torch::rand({400, 3}, torch::kFloat64) * 4.0});
    auto feats = torch::randn({1000, channels}, torch::kFloat64).to(dtype);
    AdaptiveVoxelConfig cfg;
    cfg.base_voxel_size = {2.0, 2.0, 2.0};
    cfg.max_depth = 3;
    auto t = AdaptiveSparseTensor::from_points(pts, feats, cfg);
    t.refine(AxisAlignedRegion{{0, 0, 0}, {4, 4, 4}}, 3);
    return t;
}

// The formulation before batching: gather, weight, GEMM and scatter-add per kernel slot.
at::Tensor reference_apply(const at::Tensor& x, const at::Tensor& weight,
                           const AdaptiveKernelMap& m, bool transpose, int64_t n_out,
                           bool weighted) {
    const auto dev = x.device();
    auto from = (transpose ? m.out_index : m.in_index).to(dev);
    auto to = (transpose ? m.in_index : m.out_index).to(dev);
    at::Tensor w;
    if (weighted) w = m.weight.to(dev).to(x.scalar_type());
    auto wt = weight.to(x.scalar_type());
    auto y = torch::zeros({n_out, weight.size(2)}, x.options());
    for (int64_t k = 0; k < m.num_kernels; ++k) {
        const int64_t b = m.kernel_offsets[k], e = m.kernel_offsets[k + 1];
        if (b == e) continue;
        auto xs = x.index_select(0, from.slice(0, b, e));
        if (weighted) xs = xs * w.slice(0, b, e).unsqueeze(1);
        y.index_add_(0, to.slice(0, b, e), xs.matmul(wt[k]));
    }
    return y;
}

void check_against_reference(torch::Device dev, torch::ScalarType dtype, double tol) {
    const auto modes = {AdaptiveConvMode::CenterSampled, AdaptiveConvMode::VolumeWeighted,
                        AdaptiveConvMode::OverlapWeighted};
    auto base = mixed_tensor(4, dtype);
    for (auto mode : modes) {
        for (int shift : {0, -1}) {
            AdaptiveKernelConfig kc;
            kc.mode = mode;
            kc.output_level_shift = shift;
            const bool weighted = mode != AdaptiveConvMode::CenterSampled;
            KernelMapCache::instance().clear();
            auto entry = KernelMapCache::instance().get_or_build(base, kc);
            const int64_t n_out = entry.out_topology->size();

            // Forward convolution, with gradients w.r.t. features and weight.
            auto x = base.features().to(dev).detach().requires_grad_(true);
            auto w = torch::randn({27, 4, 5}, torch::kFloat64).to(dtype).to(dev).requires_grad_(true);
            auto y = adaptive_convolution(base.with_features(x), w, kc).features();
            auto x2 = x.detach().clone().requires_grad_(true);
            auto w2 = w.detach().clone().requires_grad_(true);
            auto y2 = reference_apply(x2, w2, entry.map, false, n_out, weighted);
            CHECK_EQ(y.size(0), n_out);
            CHECK_ALLCLOSE(y, y2, tol, tol);
            auto g = torch::randn_like(y);
            (y * g).sum().backward();
            (y2 * g).sum().backward();
            CHECK_ALLCLOSE(x.grad(), x2.grad(), tol, tol);
            CHECK_ALLCLOSE(w.grad(), w2.grad(), tol, tol);

            // Transposed convolution back onto the input leaves.
            auto coarse = AdaptiveSparseTensor::from_topology(
                entry.out_topology, torch::randn({n_out, 5}, torch::kFloat64).to(dtype).to(dev));
            auto wtr = torch::randn({27, 5, 4}, torch::kFloat64).to(dtype).to(dev);
            auto up = adaptive_convolution_transpose(coarse, wtr, kc, base.to(dev)).features();
            auto up2 = reference_apply(coarse.features(), wtr, entry.map, true, base.size(),
                                       weighted);
            CHECK_ALLCLOSE(up, up2, tol, tol);
        }
    }
    KernelMapCache::instance().clear();
}

}  // namespace

TEST(device_kmap_cpu_matches_per_slot_reference) {
    check_against_reference(torch::kCPU, torch::kFloat64, 1e-12);
    check_against_reference(torch::kCPU, torch::kFloat32, 1e-5);
}

TEST(device_kmap_cuda_matches_per_slot_reference) {
    if (!torch::cuda::is_available()) return;
    check_against_reference(torch::kCUDA, torch::kFloat64, 1e-12);
    check_against_reference(torch::kCUDA, torch::kFloat32, 1e-4);
}

TEST(device_kmap_cache_uploads_once_per_device_and_dtype) {
    auto t = mixed_tensor(2, torch::kFloat64);
    AdaptiveKernelConfig kc;
    kc.mode = AdaptiveConvMode::OverlapWeighted;
    KernelMapCache::instance().clear();
    auto e1 = KernelMapCache::instance().get_or_build(t, kc);
    auto e2 = KernelMapCache::instance().get_or_build(t, kc);
    CHECK(e1.device.get() == e2.device.get());  // copies of one entry share the cache

    auto a = e1.device->get(e1.map, torch::kCPU, torch::kFloat32, true);
    auto b = e2.device->get(e2.map, torch::kCPU, torch::kFloat32, true);
    CHECK(a.weight.data_ptr() == b.weight.data_ptr());
    CHECK(a.in_index.data_ptr() == b.in_index.data_ptr());
    auto c = e1.device->get(e1.map, torch::kCPU, torch::kFloat64, true);
    CHECK(c.weight.scalar_type() == torch::kFloat64);
    CHECK(c.weight.data_ptr() != a.weight.data_ptr());
    CHECK(torch::equal(c.weight, e1.map.weight));
    auto n = e1.device->get(e1.map, torch::kCPU, torch::kFloat64, false);
    CHECK(!n.weight.defined());

    // Convolving twice reuses the same arrays and gives identical results on the CPU.
    auto w = torch::randn({27, 2, 3}, torch::kFloat64);
    auto y1 = adaptive_convolution(t, w, kc).features();
    auto y2 = adaptive_convolution(t, w, kc).features();
    CHECK(torch::equal(y1, y2));

    if (torch::cuda::is_available()) {
        auto g1 = e1.device->get(e1.map, torch::kCUDA, torch::kFloat32, true);
        auto g2 = e1.device->get(e1.map, torch::kCUDA, torch::kFloat32, true);
        CHECK(g1.in_index.is_cuda());
        CHECK(g1.in_index.data_ptr() == g2.in_index.data_ptr());
        CHECK(g1.weight.data_ptr() == g2.weight.data_ptr());
    }
    KernelMapCache::instance().clear();
}

TEST(device_kmap_empty_map_gives_zeros) {
    // An empty tensor has no pairs at all.
    AdaptiveVoxelConfig cfg;
    cfg.base_voxel_size = {1.0, 1.0, 1.0};
    cfg.max_depth = 2;
    auto t = AdaptiveSparseTensor::from_voxels(torch::empty({0, 3}, torch::kInt64),
                                               torch::empty({0}, torch::kInt32),
                                               torch::empty({0, 2}, torch::kFloat64), cfg);
    AdaptiveKernelConfig kc;
    auto w = torch::randn({27, 2, 3}, torch::kFloat64).requires_grad_(true);
    auto y = adaptive_convolution(t, w, kc);
    CHECK_EQ(y.size(), int64_t(0));
    CHECK_EQ(y.features().size(1), int64_t(3));
    // Still part of the graph: backward gives zero weight gradients, no error.
    y.features().sum().backward();
    CHECK(w.grad().defined());
    CHECK_EQ(w.grad().abs().sum().item<double>(), 0.0);
}

TEST(device_kmap_cuda_deterministic_mode_is_bit_reproducible) {
    if (!torch::cuda::is_available()) return;
    auto t = mixed_tensor(8, torch::kFloat32).to(torch::kCUDA);
    AdaptiveKernelConfig kc;
    kc.mode = AdaptiveConvMode::OverlapWeighted;
    auto w = torch::randn({27, 8, 16}, torch::kFloat32).to(torch::kCUDA);
    auto& ctx = at::globalContext();
    const bool was = ctx.deterministicAlgorithms();
    const bool warn = ctx.deterministicAlgorithmsWarnOnly();
    ctx.setDeterministicAlgorithms(true, false);
    auto y1 = adaptive_convolution(t, w, kc).features();
    auto y2 = adaptive_convolution(t, w, kc).features();
    ctx.setDeterministicAlgorithms(was, warn);
    CHECK(torch::equal(y1, y2));
    // Without it, results agree to float32 rounding (atomics may reorder the additions).
    auto y3 = adaptive_convolution(t, w, kc).features();
    CHECK_ALLCLOSE(y1, y3, 1e-5, 1e-5);
    KernelMapCache::instance().clear();
}
