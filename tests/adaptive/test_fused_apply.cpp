// Fused single-launch CUDA apply_map (fused_apply_cuda.cu): forward, input and weight
// gradients must match the per-slot reference and the unfused CUDA path within float
// accumulation tolerances, pass a float64 finite-difference check, keep higher-order
// gradients, and give way to the unfused path in deterministic mode.
// Tolerances: float64 1e-10 (atomics only reorder the sums). float32 results are judged
// against float64 on the same inputs (check_against): at most twice the per-slot path's error.
#include <vector>

#include "MinkowskiEngine/adaptive/adaptive.hpp"
#include "MinkowskiEngine/adaptive/adaptive_topology.hpp"
#include "utils/test_framework.hpp"

using namespace minkowski::adaptive;
namespace ad = minkowski::adaptive::detail;

namespace {

void require_cuda() {
    if (!torch::cuda::is_available()) throw mink_test::Skip("CUDA not available");
}

// Sets the fused-kernel switch for one scope.
struct Fused {
    bool saved;
    explicit Fused(bool on) : saved(ad::cuda_apply_map_fused()) {
        ad::cuda_apply_map_fused() = on;
    }
    ~Fused() { ad::cuda_apply_map_fused() = saved; }
};

AdaptiveSparseTensor mixed_tensor(int64_t channels, torch::ScalarType dtype) {
    torch::manual_seed(5);
    auto pts = torch::cat({torch::rand({500, 3}, torch::kFloat64) * 32.0,
                           torch::rand({300, 3}, torch::kFloat64) * 4.0});
    auto feats = torch::randn({800, channels}, torch::kFloat64).to(dtype);
    AdaptiveVoxelConfig cfg;
    cfg.base_voxel_size = {2.0, 2.0, 2.0};
    cfg.max_depth = 3;
    auto t = AdaptiveSparseTensor::from_points(pts, feats, cfg);
    t.refine(AxisAlignedRegion{{0, 0, 0}, {4, 4, 4}}, 3);
    return t;
}

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

struct Result {
    at::Tensor y, gx, gw;
};

// Convolution (or its transpose) of `t` with features x and weight w, plus the gradients of
// sum(y * g) with g drawn from a fixed seed.
Result run(const AdaptiveSparseTensor& t, const at::Tensor& x0, const at::Tensor& w0,
           const AdaptiveKernelConfig& kc, bool transpose) {
    auto x = x0.detach().clone().requires_grad_(true);
    auto w = w0.detach().clone().requires_grad_(true);
    at::Tensor y;
    if (!transpose) {
        y = adaptive_convolution(t.with_features(x), w, kc).features();
    } else {
        auto entry = KernelMapCache::instance().get_or_build(t, kc);
        auto coarse = AdaptiveSparseTensor::from_topology(entry.out_topology, x);
        y = adaptive_convolution_transpose(coarse, w, kc, t).features();
    }
    torch::manual_seed(11);
    auto g = torch::randn(y.sizes(), torch::kFloat64).to(y.dtype()).to(y.device());
    (y * g).sum().backward();
    return {y.detach(), x.grad(), w.grad()};
}

const AdaptiveConvMode kModes[] = {AdaptiveConvMode::CenterSampled,
                                   AdaptiveConvMode::VolumeWeighted,
                                   AdaptiveConvMode::OverlapWeighted};

// float64: fused and reference agree to 1e-10. float32: both are compared with the float64
// result on the same (float32-rounded) inputs, and the fused error may be at most twice the
// per-slot path's own float32 error (or 1e-6 of the largest value): the atomics change the
// summation order, not the accuracy.
void check_against(const at::Tensor& got, const at::Tensor& ref, const at::Tensor& truth,
                   const char* what) {
    if (got.scalar_type() == torch::kFloat64) {
        CHECK_ALLCLOSE(got, ref, 1e-10, 1e-10);
        return;
    }
    const auto t = truth.cpu();
    const double scale = t.abs().max().item<double>();
    const double e_got = (got.cpu().to(torch::kFloat64) - t).abs().max().item<double>();
    const double e_ref = (ref.cpu().to(torch::kFloat64) - t).abs().max().item<double>();
    if (!(e_got <= std::max(2.0 * e_ref, 1e-6 * scale)))
        FAIL_AT(what << ": fused float32 error " << e_got << " vs per-slot " << e_ref
                     << " (max |value| " << scale << ")");
}

// Reference convolution (per slot) of x/w with the loss sum(y * g), g from seed 11; returns
// y and the gradients.
Result reference_run(const AdaptiveKernelMap& m, const at::Tensor& x0, const at::Tensor& w0,
                     bool transpose, int64_t n_out, bool weighted, torch::ScalarType gdtype) {
    auto x = x0.detach().clone().requires_grad_(true);
    auto w = w0.detach().clone().requires_grad_(true);
    auto y = reference_apply(x, w, m, transpose, n_out, weighted);
    torch::manual_seed(11);
    auto g = torch::randn(y.sizes(), torch::kFloat64).to(gdtype).to(y.dtype()).to(y.device());
    (y * g).sum().backward();
    return {y.detach(), x.grad(), w.grad()};
}

}  // namespace

TEST(fused_apply_cuda_matches_per_slot_reference) {
    require_cuda();
    Fused on(true);
    // Channel pairs covering every kernel (8 and 16 tiles, register-blocked 64 x 64) and
    // counts that are not multiples of the tile.
    const std::vector<std::pair<int64_t, int64_t>> channels = {
        {4, 5}, {3, 70}, {8, 16}, {24, 32}, {32, 64}, {40, 70}};
    for (auto dtype : {torch::kFloat64, torch::kFloat32}) {
        for (auto [cin, cout] : channels) {
            auto base = mixed_tensor(cin, dtype).to(torch::kCUDA);
            for (auto mode : kModes) {
                for (int shift : {0, -1}) {
                    AdaptiveKernelConfig kc;
                    kc.mode = mode;
                    kc.output_level_shift = shift;
                    const bool weighted = mode != AdaptiveConvMode::CenterSampled;
                    KernelMapCache::instance().clear();
                    auto entry = KernelMapCache::instance().get_or_build(base, kc);
                    const int64_t n_out = entry.out_topology->size();
                    auto compare = [&](const at::Tensor& x, const at::Tensor& w, bool transpose,
                                       int64_t rows_out) {
                        auto got = run(base, x, w, kc, transpose);
                        auto ref = reference_run(entry.map, x, w, transpose, rows_out, weighted,
                                                 dtype);
                        auto truth = reference_run(entry.map, x.to(torch::kFloat64),
                                                   w.to(torch::kFloat64), transpose, rows_out,
                                                   weighted, dtype);
                        check_against(got.y, ref.y, truth.y, "y");
                        check_against(got.gx, ref.gx, truth.gx, "grad x");
                        check_against(got.gw, ref.gw, truth.gw, "grad w");
                    };
                    torch::manual_seed(7);
                    auto w = torch::randn({27, cin, cout}, torch::kFloat64).to(dtype).to(torch::kCUDA);
                    compare(base.features(), w, false, n_out);
                    // Transposed: coarse [n_out, cout] features back onto the input leaves.
                    torch::manual_seed(9);
                    auto xt = torch::randn({n_out, cout}, torch::kFloat64).to(dtype).to(torch::kCUDA);
                    auto wt = torch::randn({27, cout, cin}, torch::kFloat64).to(dtype).to(torch::kCUDA);
                    compare(xt, wt, true, base.size());
                }
            }
        }
    }
    KernelMapCache::instance().clear();
}

TEST(fused_apply_cuda_matches_unfused_path_and_cpu) {
    require_cuda();
    auto base_cpu = mixed_tensor(8, torch::kFloat64);
    auto base = base_cpu.to(torch::kCUDA);
    for (auto mode : kModes) {
        AdaptiveKernelConfig kc;
        kc.mode = mode;
        KernelMapCache::instance().clear();
        torch::manual_seed(4);
        auto w = torch::randn({27, 8, 16}, torch::kFloat64);
        Result fused, unfused;
        {
            Fused on(true);
            fused = run(base, base.features(), w.to(torch::kCUDA), kc, false);
        }
        {
            Fused off(false);
            unfused = run(base, base.features(), w.to(torch::kCUDA), kc, false);
        }
        auto cpu = run(base_cpu, base_cpu.features(), w, kc, false);
        CHECK_ALLCLOSE(fused.y, unfused.y, 1e-10, 1e-10);
        CHECK_ALLCLOSE(fused.gx, unfused.gx, 1e-10, 1e-10);
        CHECK_ALLCLOSE(fused.gw, unfused.gw, 1e-10, 1e-10);
        CHECK_ALLCLOSE(fused.y.cpu(), cpu.y, 1e-10, 1e-10);
        CHECK_ALLCLOSE(fused.gx.cpu(), cpu.gx, 1e-10, 1e-10);
        CHECK_ALLCLOSE(fused.gw.cpu(), cpu.gw, 1e-10, 1e-10);
    }
    KernelMapCache::instance().clear();
}

TEST(fused_apply_cuda_finite_difference_gradients) {
    require_cuda();
    Fused on(true);
    // Small tensor so central differences over a sample of entries stay cheap.
    torch::manual_seed(2);
    auto pts = torch::rand({60, 3}, torch::kFloat64) * 8.0;
    AdaptiveVoxelConfig cfg;
    cfg.base_voxel_size = {2.0, 2.0, 2.0};
    cfg.max_depth = 2;
    auto t = AdaptiveSparseTensor::from_points(pts, torch::randn({60, 3}, torch::kFloat64), cfg);
    t.refine(AxisAlignedRegion{{0, 0, 0}, {4, 4, 4}}, 2);
    t = t.to(torch::kCUDA);
    for (auto mode : kModes) {
        AdaptiveKernelConfig kc;
        kc.mode = mode;
        auto x = t.features().detach().clone();
        auto w = torch::randn({27, 3, 2}, torch::kFloat64).to(torch::kCUDA);
        auto analytic = run(t, x, w, kc, false);
        torch::manual_seed(11);
        auto g = torch::randn(analytic.y.sizes(), torch::kFloat64).to(torch::kCUDA);
        auto loss = [&](const at::Tensor& xv, const at::Tensor& wv) {
            return (adaptive_convolution(t.with_features(xv), wv, kc).features() * g)
                .sum()
                .item<double>();
        };
        const double eps = 1e-6;
        auto xf = x.view(-1), wf = w.view(-1);
        for (int64_t i = 0; i < xf.numel(); i += std::max<int64_t>(1, xf.numel() / 40)) {
            auto xp = x.clone(), xm = x.clone();
            xp.view(-1)[i] += eps;
            xm.view(-1)[i] -= eps;
            const double num = (loss(xp, w) - loss(xm, w)) / (2 * eps);
            CHECK_NEAR(analytic.gx.view(-1)[i].item<double>(), num, 1e-6);
        }
        for (int64_t i = 0; i < wf.numel(); i += 7) {
            auto wp = w.clone(), wm = w.clone();
            wp.view(-1)[i] += eps;
            wm.view(-1)[i] -= eps;
            const double num = (loss(x, wp) - loss(x, wm)) / (2 * eps);
            CHECK_NEAR(analytic.gw.view(-1)[i].item<double>(), num, 1e-6);
        }
    }
    KernelMapCache::instance().clear();
}

TEST(fused_apply_cuda_second_order_gradients_match_unfused) {
    require_cuda();
    auto base = mixed_tensor(4, torch::kFloat64).to(torch::kCUDA);
    AdaptiveKernelConfig kc;
    kc.mode = AdaptiveConvMode::VolumeWeighted;
    torch::manual_seed(6);
    auto w0 = torch::randn({27, 4, 5}, torch::kFloat64).to(torch::kCUDA);
    // d/d(x, w) of |d loss / d x|^2: needs the backward pass itself to be differentiable.
    auto second = [&](bool fused) {
        Fused f(fused);
        auto x = base.features().detach().clone().requires_grad_(true);
        auto w = w0.clone().requires_grad_(true);
        auto y = adaptive_convolution(base.with_features(x), w, kc).features();
        auto gx = torch::autograd::grad({(y * y).sum()}, {x}, {}, /*retain_graph=*/true,
                                        /*create_graph=*/true)[0];
        gx.pow(2).sum().backward();
        return std::make_pair(x.grad(), w.grad());
    };
    auto a = second(true);
    auto b = second(false);
    CHECK_ALLCLOSE(a.first, b.first, 1e-9, 1e-9);
    CHECK_ALLCLOSE(a.second, b.second, 1e-9, 1e-9);
    KernelMapCache::instance().clear();
}

TEST(fused_apply_cuda_edge_cases) {
    require_cuda();
    Fused on(true);
    // Empty tensor: no pairs, zero rows out, zero weight gradient.
    AdaptiveVoxelConfig cfg;
    cfg.base_voxel_size = {1.0, 1.0, 1.0};
    cfg.max_depth = 2;
    auto empty = AdaptiveSparseTensor::from_voxels(torch::empty({0, 3}, torch::kInt64),
                                                   torch::empty({0}, torch::kInt32),
                                                   torch::empty({0, 2}, torch::kFloat32), cfg)
                     .to(torch::kCUDA);
    auto w = torch::randn({27, 2, 3}, torch::kFloat32).to(torch::kCUDA).requires_grad_(true);
    auto y = adaptive_convolution(empty, w, AdaptiveKernelConfig{});
    CHECK_EQ(y.size(), int64_t(0));
    CHECK_EQ(y.features().size(1), int64_t(3));
    y.features().sum().backward();
    CHECK(w.grad().defined());
    CHECK_EQ(w.grad().abs().sum().item<double>(), 0.0);

    // Kernel size 1: a single slot.
    auto base = mixed_tensor(5, torch::kFloat64).to(torch::kCUDA);
    AdaptiveKernelConfig k1;
    k1.kernel_size = 1;
    auto w1 = torch::randn({1, 5, 4}, torch::kFloat64).to(torch::kCUDA);
    auto entry = KernelMapCache::instance().get_or_build(base, k1);
    auto r = run(base, base.features(), w1, k1, false);
    auto ref = reference_apply(base.features(), w1, entry.map, false, entry.out_topology->size(),
                               false);
    CHECK_ALLCLOSE(r.y, ref, 1e-10, 1e-10);

    // Non-contiguous features (a column slice) and a float64 weight with float32 features
    // (the weight gradient comes back in the weight's own dtype).
    auto wide = torch::randn({base.size(), 10}, torch::kFloat32).to(torch::kCUDA);
    auto xs = wide.slice(1, 2, 7);
    CHECK(!xs.is_contiguous());
    auto wd = torch::randn({27, 5, 6}, torch::kFloat64).to(torch::kCUDA).requires_grad_(true);
    AdaptiveKernelConfig kc;
    auto ys = adaptive_convolution(base.with_features(xs), wd, kc).features();
    auto entry3 = KernelMapCache::instance().get_or_build(base, kc);
    auto yr = reference_apply(xs.contiguous(), wd.detach(), entry3.map, false,
                              entry3.out_topology->size(), false);
    CHECK_ALLCLOSE(ys, yr, 1e-4, 1e-4);
    ys.sum().backward();
    CHECK(wd.grad().scalar_type() == torch::kFloat64);
    KernelMapCache::instance().clear();
}

TEST(fused_apply_cuda_more_tiles_than_one_grid_row) {
    require_cuda();
    // > 65535 tiles (8-row tiles for 3 -> 5 channels), the gridDim.y limit a per-slot-on-y
    // layout would hit; checked against the unfused path in float64.
    torch::manual_seed(0);
    auto pts = torch::cat({torch::rand({10000, 3}, torch::kFloat64) * 256.0,
                           torch::rand({10000, 3}, torch::kFloat64) * 16.0});
    AdaptiveVoxelConfig cfg;
    cfg.base_voxel_size = {8.0, 8.0, 8.0};
    cfg.min_voxel_size = {0.5, 0.5, 0.5};
    cfg.max_depth = 4;
    auto t = AdaptiveSparseTensor::from_points(pts, torch::randn({20000, 3}, torch::kFloat64), cfg);
    t.refine(AxisAlignedRegion{{0, 0, 0}, {16, 16, 16}}, 4);
    t = t.to(torch::kCUDA);
    AdaptiveKernelConfig kc;
    KernelMapCache::instance().clear();
    auto entry = KernelMapCache::instance().get_or_build(t, kc);
    auto tiles = entry.device->tiles(entry.map, torch::Device(torch::kCUDA), 8);
    CHECK(tiles.size(0) > 65535);
    auto w = torch::randn({27, 3, 5}, torch::kFloat64).to(torch::kCUDA);
    Result a, b;
    {
        Fused on(true);
        a = run(t, t.features(), w, kc, false);
    }
    {
        Fused off(false);
        b = run(t, t.features(), w, kc, false);
    }
    CHECK_ALLCLOSE(a.y, b.y, 1e-10, 1e-10);
    CHECK_ALLCLOSE(a.gx, b.gx, 1e-10, 1e-10);
    CHECK_ALLCLOSE(a.gw, b.gw, 1e-9, 1e-9);
    KernelMapCache::instance().clear();
}

TEST(fused_apply_tile_table_covers_each_slot_once) {
    auto t = mixed_tensor(2, torch::kFloat64);
    AdaptiveKernelConfig kc;
    kc.mode = AdaptiveConvMode::OverlapWeighted;
    KernelMapCache::instance().clear();
    auto e = KernelMapCache::instance().get_or_build(t, kc);
    const auto& off = e.map.kernel_offsets;
    for (int64_t rows : {int64_t(1), int64_t(8), int64_t(256), int64_t(1) << 40}) {
        auto tiles = e.device->tiles(e.map, torch::kCPU, rows);
        CHECK(tiles.data_ptr() == e.device->tiles(e.map, torch::kCPU, rows).data_ptr());
        auto acc = tiles.accessor<int64_t, 2>();
        int64_t next = 0, prev_k = -1;
        for (int64_t i = 0; i < tiles.size(0); ++i) {
            const int64_t b = acc[i][0], en = acc[i][1], k = acc[i][2];
            CHECK(k >= prev_k);
            if (k != prev_k) {
                CHECK_EQ(b, off[k]);  // empty slots have no tiles
                next = b;
            }
            CHECK_EQ(b, next);
            CHECK(en > b && en - b <= rows && en <= off[k + 1]);
            next = en;
            prev_k = k;
        }
        int64_t covered = 0;
        for (int64_t i = 0; i < tiles.size(0); ++i) covered += acc[i][1] - acc[i][0];
        CHECK_EQ(covered, off.back());
    }
    KernelMapCache::instance().clear();
}

TEST(fused_apply_cuda_deterministic_mode_uses_unfused_path) {
    require_cuda();
    auto t = mixed_tensor(8, torch::kFloat32).to(torch::kCUDA);
    AdaptiveKernelConfig kc;
    kc.mode = AdaptiveConvMode::OverlapWeighted;
    auto w = torch::randn({27, 8, 16}, torch::kFloat32).to(torch::kCUDA);
    auto& ctx = at::globalContext();
    const bool was = ctx.deterministicAlgorithms();
    const bool warn = ctx.deterministicAlgorithmsWarnOnly();
    ctx.setDeterministicAlgorithms(true, false);
    at::Tensor y_fused_flag, y_unfused;
    {
        Fused on(true);
        y_fused_flag = adaptive_convolution(t, w, kc).features();
    }
    {
        Fused off(false);
        y_unfused = adaptive_convolution(t, w, kc).features();
    }
    ctx.setDeterministicAlgorithms(was, warn);
    CHECK(torch::equal(y_fused_flag, y_unfused));
    KernelMapCache::instance().clear();
}
