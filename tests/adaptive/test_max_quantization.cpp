// MAX_POOL quantization of the sparse engine (SparseTensor, TensorField::sparse): a voxel whose
// maximum is exactly 0 keeps the whole gradient (see gradient_max_pool_at_zero_keeps_unit_gradient
// in test_gradients.cpp for the adaptive pooling version of the same fix). Kept in its own file:
// the sparse-engine headers pull in <windows.h> on CUDA builds, whose `far`/`near` macros break
// unrelated test code.
#include "MinkowskiEngine/sparse_tensor.hpp"
#include "MinkowskiEngine/tensor_field.hpp"
#include "utils/test_framework.hpp"

#include <torch/torch.h>

TEST(gradient_sparse_max_pool_quantization_at_zero_keeps_unit_gradient) {
    // Points 0 and 1 share voxel (0,0,0) with values 0 and -1; point 2 is alone with 2.
    auto coords = torch::tensor({{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 1, 0, 0}}, torch::kInt32);
    const auto f0 = torch::tensor({{0.0}, {-1.0}, {2.0}}, torch::kFloat64);
    for (const at::Device dev : {at::Device(at::kCPU), at::Device(at::kCUDA)}) {
        if (dev.is_cuda() && !torch::cuda::is_available())
            continue;
        {
            auto f = f0.to(dev).detach().clone().requires_grad_(true);
            minkowski::SparseTensor s(f, coords.to(dev), nullptr,
                                      minkowski::SparseTensorQuantizationMode::MAX_POOL);
            CHECK_EQ(s.F().size(0), int64_t(2));
            CHECK_NEAR(s.F().sum().item<double>(), 2.0, 0.0);
            s.F().sum().backward();
            auto g = f.grad().cpu();
            CHECK_NEAR(g[0][0].item<double>(), 1.0, 0.0); // the voxel's maximum is exactly 0
            CHECK_NEAR(g[1][0].item<double>(), 0.0, 0.0);
            CHECK_NEAR(g[2][0].item<double>(), 1.0, 0.0);
        }
        {
            auto f = f0.to(dev).detach().clone().requires_grad_(true);
            minkowski::TensorField field(f, coords.to(torch::kFloat32).to(dev)); // field coordinates are float
            auto s = field.sparse({}, c10::nullopt, minkowski::SparseTensorQuantizationMode::MAX_POOL);
            CHECK_EQ(s.F().size(0), int64_t(2));
            s.F().sum().backward();
            auto g = f.grad().cpu();
            CHECK_NEAR(g[0][0].item<double>(), 1.0, 0.0);
            CHECK_NEAR(g[2][0].item<double>(), 1.0, 0.0);
            CHECK_NEAR(g.sum().item<double>(), 2.0, 0.0);
        }
    }
}

