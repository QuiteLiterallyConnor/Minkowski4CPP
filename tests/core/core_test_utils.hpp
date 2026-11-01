// Shared helpers for the core engine tests (tests/core).
#ifndef MINK_CORE_TEST_UTILS_HPP
#define MINK_CORE_TEST_UTILS_HPP

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "MinkowskiEngine/minkowski.hpp"
#include "utils/test_framework.hpp"

namespace core_test {

// Unique random coordinates [batches * n_per_batch, 1 + D] (int32, batch index first) with
// every spatial coordinate in [0, extent).
inline at::Tensor random_coords(int64_t n_per_batch, int64_t extent, int64_t batches, int64_t D,
                                uint64_t seed) {
    torch::manual_seed(seed);
    int64_t cells = 1;
    for (int64_t d = 0; d < D; ++d) cells *= extent;
    std::vector<at::Tensor> parts;
    for (int64_t b = 0; b < batches; ++b) {
        auto lin = torch::randperm(cells, torch::kInt64).slice(0, 0, n_per_batch);
        auto c = torch::empty({n_per_batch, 1 + D}, torch::kInt64);
        c.select(1, 0).fill_(b);
        for (int64_t d = D - 1; d >= 0; --d) {
            c.select(1, 1 + d).copy_(lin.remainder(extent));
            lin = lin.div(extent, "floor");
        }
        parts.push_back(c);
    }
    return torch::cat(parts).to(torch::kInt32);
}

// Scatter sparse rows into a dense [B, C, L, L, L] grid; coordinates are divided by `stride`.
// Differentiable with respect to `feats`.
inline at::Tensor to_dense(const at::Tensor& coords, const at::Tensor& feats, int64_t B, int64_t L,
                           int64_t stride) {
    auto c = coords.to(torch::kInt64);
    c10::List<std::optional<at::Tensor>> idx;
    idx.push_back(c.select(1, 0));
    for (int d = 1; d <= 3; ++d) idx.push_back(c.select(1, d).div(stride, "floor"));
    auto dense = torch::zeros({B, L, L, L, feats.size(1)}, feats.options());
    return at::index_put(dense, idx, feats, false).permute({0, 4, 1, 2, 3});
}

// Read rows of a dense [B, C, X, Y, Z] grid at `coords / stride`.
inline at::Tensor gather_dense(const at::Tensor& dense, const at::Tensor& coords, int64_t stride) {
    auto c = coords.to(torch::kInt64);
    using at::indexing::TensorIndex;
    return dense.permute({0, 2, 3, 4, 1})
        .index({TensorIndex(c.select(1, 0)), TensorIndex(c.select(1, 1).div(stride, "floor")),
                TensorIndex(c.select(1, 2).div(stride, "floor")),
                TensorIndex(c.select(1, 3).div(stride, "floor"))});
}

// MinkowskiEngine kernel [K^3, Cin, Cout] (x fastest: k = ix + K*iy + K*K*iz) as a dense
// conv3d weight [Cout, Cin, K, K, K] (cross-correlation, dims ordered x, y, z).
inline at::Tensor me_kernel_to_conv3d(const at::Tensor& w, int64_t K = 3) {
    return w.view({K, K, K, w.size(1), w.size(2)}).permute({4, 3, 2, 1, 0});
}

// Same kernel as a conv_transpose3d weight [Cin, Cout, K, K, K].
inline at::Tensor me_kernel_to_conv_transpose3d(const at::Tensor& w, int64_t K = 3) {
    return w.view({K, K, K, w.size(1), w.size(2)}).permute({3, 4, 2, 1, 0});
}

// Rows sorted by coordinate (batch, x, y, z, ...) so tensors with different row orders compare.
inline at::Tensor sort_rows_by_coords(const at::Tensor& coords, const at::Tensor& feats) {
    auto c = coords.to(torch::kCPU).to(torch::kInt64);
    auto key = torch::zeros({c.size(0)}, torch::kInt64);
    for (int64_t d = 0; d < c.size(1); ++d) key = key * 4096 + (c.select(1, d) + 2048);
    auto order = std::get<1>(key.sort());
    return feats.to(torch::kCPU).index_select(0, order);
}

// Central finite-difference check of every element of `params` (contiguous fp64 leaf tensors)
// against autograd, using the scalar loss sum(forward() * R) for a fixed random R.
inline void gradcheck(const std::function<at::Tensor()>& forward, std::vector<at::Tensor> params,
                      double eps = 1e-6, double tol = 1e-6, uint64_t seed = 1234) {
    for (auto& p : params) {
        if (p.grad().defined()) p.mutable_grad().zero_();
    }
    auto out = forward();
    torch::manual_seed(seed);
    auto R = torch::randn_like(out).detach();
    (out * R).sum().backward();
    for (size_t pi = 0; pi < params.size(); ++pi) {
        auto& p = params[pi];
        CHECK(p.is_contiguous() && p.scalar_type() == torch::kFloat64);
        auto analytic = p.grad().detach().clone();
        auto numeric = torch::zeros_like(analytic);
        torch::NoGradGuard ng;
        double* data = p.data_ptr<double>();
        double* num = numeric.data_ptr<double>();
        for (int64_t i = 0; i < p.numel(); ++i) {
            const double orig = data[i];
            data[i] = orig + eps;
            const double fp = (forward() * R).sum().item<double>();
            data[i] = orig - eps;
            const double fm = (forward() * R).sum().item<double>();
            data[i] = orig;
            num[i] = (fp - fm) / (2 * eps);
        }
        if (!torch::allclose(analytic, numeric, tol, tol))
            FAIL_AT("gradcheck: parameter " << pi << " max |analytic - numeric| = "
                                            << (analytic - numeric).abs().max().item<double>());
    }
}

}  // namespace core_test

#endif
