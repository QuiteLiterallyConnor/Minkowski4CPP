// CPU and CUDA backends must agree on outputs and gradients (rows matched by coordinate).
// Skipped in CPU_ONLY builds and when no CUDA device is available.
#include "core_test_utils.hpp"

using namespace minkowski;

namespace {

#ifndef CPU_ONLY
// Loss weights that depend only on (coordinate, channel), so both devices see the same loss
// regardless of row order.
at::Tensor coordinate_weights(const at::Tensor& coords, int64_t channels) {
    auto c = coords.to(torch::kFloat64);
    auto phase = 0.37 * c.select(1, 0) + 1.3 * c.select(1, 1) + 0.7 * c.select(1, 2) +
                 2.1 * c.select(1, 3);
    auto ch = torch::arange(channels, c.options()) * 0.5;
    return torch::sin(phase.unsqueeze(1) + ch.unsqueeze(0));
}

struct Run {
    at::Tensor out_sorted, grad_feats, grad_kernel;
};

// `make` builds the module under test (fresh, deterministic weights for a given seed).
template <typename MakeModule>
Run run_on(torch::Device device, const at::Tensor& coords, const at::Tensor& feats0,
           MakeModule make, bool has_kernel) {
    auto module = make();
    module->to(device);
    auto feats = feats0.to(device).detach().requires_grad_(true);
    SparseTensor x(feats, coords.to(device));
    auto out = module->forward(x);
    auto F = out.F();
    auto C = out.C();
    (F * coordinate_weights(C, F.size(1)).to(F.dtype()).to(device)).sum().backward();
    Run r;
    r.out_sorted = core_test::sort_rows_by_coords(C, F.detach());
    r.grad_feats = feats.grad().to(torch::kCPU);
    if (has_kernel) {
        for (auto& p : module->parameters())
            if (p.dim() == 3) r.grad_kernel = p.grad().to(torch::kCPU);
    }
    return r;
}

template <typename MakeModule>
void compare_devices(MakeModule make, bool has_kernel, int64_t cin, uint64_t seed,
                     double tol = 1e-4) {
    if (!torch::cuda::is_available()) throw mink_test::Skip("CUDA device not available");
    auto coords = core_test::random_coords(300, 12, 2, 3, seed);
    torch::manual_seed(seed);
    auto feats = torch::randn({coords.size(0), cin}, torch::kFloat32);
    auto cpu = run_on(torch::kCPU, coords, feats, make, has_kernel);
    auto gpu = run_on(torch::kCUDA, coords, feats, make, has_kernel);
    CHECK_EQ(cpu.out_sorted.size(0), gpu.out_sorted.size(0));
    CHECK_ALLCLOSE(cpu.out_sorted, gpu.out_sorted, tol, tol);
    CHECK_ALLCLOSE(cpu.grad_feats, gpu.grad_feats, tol, tol);
    if (has_kernel) CHECK_ALLCLOSE(cpu.grad_kernel, gpu.grad_kernel, tol, tol);
}
#endif

void skip_if_cpu_only() {
#ifdef CPU_ONLY
    throw mink_test::Skip("CPU_ONLY build");
#endif
}

}  // namespace

TEST(cpu_cuda_conv_stride1) {
    skip_if_cpu_only();
#ifndef CPU_ONLY
    compare_devices([] {
        torch::manual_seed(5);
        return MinkowskiConvolution(3, 5, 3, 1, 1, false, false, ConvolutionMode::DEFAULT, 3);
    }, true, 3, 31);
#endif
}

TEST(cpu_cuda_conv_stride2) {
    skip_if_cpu_only();
#ifndef CPU_ONLY
    compare_devices([] {
        torch::manual_seed(6);
        return MinkowskiConvolution(4, 6, 3, 2, 1, false, false, ConvolutionMode::DEFAULT, 3);
    }, true, 4, 32);
#endif
}

TEST(cpu_cuda_conv_transpose_stride1) {
    skip_if_cpu_only();
#ifndef CPU_ONLY
    compare_devices([] {
        torch::manual_seed(7);
        return MinkowskiConvolutionTranspose(5, 3, 3, 1, 1, false, false, ConvolutionMode::DEFAULT, 3);
    }, true, 5, 33);
#endif
}

TEST(cpu_cuda_avg_pooling) {
    skip_if_cpu_only();
#ifndef CPU_ONLY
    compare_devices([] { return MinkowskiAvgPooling(3, 2, 1, 3); }, false, 3, 34);
#endif
}

TEST(cpu_cuda_max_pooling) {
    skip_if_cpu_only();
#ifndef CPU_ONLY
    compare_devices([] { return MinkowskiMaxPooling(2, 2, 1, 3); }, false, 3, 35);
#endif
}

TEST(cpu_cuda_global_avg_pooling) {
    skip_if_cpu_only();
#ifndef CPU_ONLY
    compare_devices([] { return MinkowskiGlobalAvgPooling(); }, false, 4, 36);
#endif
}
