// The autograd graph must keep the coordinate manager alive: backward() has to work after every
// SparseTensor that referred to the manager is gone, e.g.
//   conv->forward(SparseTensor(feats, coords)).F().sum().backward();
// and the manager must be released together with the graph (no leak).
// Every case runs on the CPU and, when available, on CUDA (CUDA manager + CUDA kernels).
#include <memory>

#include "core_test_utils.hpp"

using namespace minkowski;

namespace {

void skip_without_cuda() {
#ifdef CPU_ONLY
    throw mink_test::Skip("CPU_ONLY build");
#else
    if (!torch::cuda::is_available()) throw mink_test::Skip("CUDA device not available");
#endif
}

// `op` maps a sparse tensor to a dense feature tensor through one autograd function. It runs on a
// tensor whose manager is only referenced by SparseTensors local to this helper; afterwards the
// gradient must match a run where the manager is still alive.
template <typename Op>
void check_graph_keeps_manager(Op op, int64_t channels, uint64_t seed, torch::Device device) {
    auto coords = core_test::random_coords(30, 5, 2, 3, seed).to(device);
    torch::manual_seed(seed);
    auto feats0 = torch::randn({coords.size(0), channels}, torch::kFloat64).to(device);

    // Reference: manager alive during backward.
    auto feats_ref = feats0.clone().requires_grad_(true);
    {
        SparseTensor x(feats_ref, coords);
        auto out = op(x);
        (out * out).sum().backward();
    }

    auto feats = feats0.clone().requires_grad_(true);
    std::weak_ptr<CoordinateManager> weak;
    at::Tensor out;
    {
        SparseTensor x(feats, coords);
        weak = x.coordinate_manager();
        out = op(x);
    }  // every SparseTensor (and its manager reference) is gone here
    if (weak.expired())
        FAIL_AT("coordinate manager destroyed while the autograd graph still needs it");
    (out * out).sum().backward();
    CHECK(feats.grad().defined());
    CHECK(feats.grad().device() == feats.device());
    // CUDA kernels accumulate with atomics, so the summation order may differ between runs.
    const double tol = device.is_cuda() ? 1e-10 : 1e-12;
    CHECK_ALLCLOSE(feats.grad().cpu(), feats_ref.grad().cpu(), tol, tol);

    out = at::Tensor();  // releasing the graph releases the manager
    if (!weak.expired()) FAIL_AT("coordinate manager leaked after the graph was released");
}

void convolution(torch::Device device) {
    MinkowskiConvolution conv(2, 3, 3, 2, 1, false, false, ConvolutionMode::DEFAULT, 3);
    conv->to(device, torch::kFloat64);
    check_graph_keeps_manager([&](const SparseTensor& x) { return conv->forward(x).F(); }, 2, 51,
                              device);
}

void convolution_transpose(torch::Device device) {
    MinkowskiConvolutionTranspose conv(2, 3, 3, 1, 1, false, false, ConvolutionMode::DEFAULT, 3);
    conv->to(device, torch::kFloat64);
    check_graph_keeps_manager([&](const SparseTensor& x) { return conv->forward(x).F(); }, 2, 52,
                              device);
}

void local_pooling(torch::Device device) {
    MinkowskiAvgPooling pool(2, 2, 1, 3);
    check_graph_keeps_manager([&](const SparseTensor& x) { return pool->forward(x).F(); }, 3, 53,
                              device);
}

void global_pooling(torch::Device device) {
    MinkowskiGlobalMaxPooling pool;
    check_graph_keeps_manager([&](const SparseTensor& x) { return pool->forward(x).F(); }, 3, 54,
                              device);
}

void broadcast(torch::Device device) {
    MinkowskiGlobalAvgPooling pool;
    MinkowskiBroadcastMultiplication mul;
    check_graph_keeps_manager(
        [&](const SparseTensor& x) {
            // Detached global features: only the broadcast function sees x's graph.
            auto glob = pool->forward(x);
            SparseTensor g(glob.F().detach(), glob.coordinate_map_key(), glob.coordinate_manager());
            return mul->forward(x, g).F();
        },
        3, 55, device);
}

void pruning(torch::Device device) {
    MinkowskiPruning prune;
    check_graph_keeps_manager(
        [&](const SparseTensor& x) {
            auto mask = torch::arange(x.F().size(0), x.F().options().dtype(torch::kInt64))
                            .remainder(3) != 0;
            return prune->forward(x, mask).F();
        },
        2, 56, device);
}

void instance_norm(torch::Device device) {
    MinkowskiInstanceNorm norm(3);
    norm->to(device, torch::kFloat64);
    check_graph_keeps_manager([&](const SparseTensor& x) { return norm->forward(x).F(); }, 3, 57,
                              device);
}

void interpolation(torch::Device device) {
    MinkowskiInterpolation interp;
    check_graph_keeps_manager(
        [&](const SparseTensor& x) {
            auto field = x.C().to(torch::kFloat64);  // batch index, then x y z
            field.slice(1, 1).add_(0.25);
            return interp->forward(x, field);
        },
        2, 58, device);
}

}  // namespace

TEST(manager_lifetime_convolution) { convolution(torch::kCPU); }
TEST(manager_lifetime_convolution_transpose) { convolution_transpose(torch::kCPU); }
TEST(manager_lifetime_local_pooling) { local_pooling(torch::kCPU); }
TEST(manager_lifetime_global_pooling) { global_pooling(torch::kCPU); }
TEST(manager_lifetime_broadcast) { broadcast(torch::kCPU); }
TEST(manager_lifetime_pruning) { pruning(torch::kCPU); }
TEST(manager_lifetime_instance_norm) { instance_norm(torch::kCPU); }
TEST(manager_lifetime_interpolation) { interpolation(torch::kCPU); }

TEST(manager_lifetime_cuda_convolution) { skip_without_cuda(); convolution(torch::kCUDA); }
TEST(manager_lifetime_cuda_convolution_transpose) { skip_without_cuda(); convolution_transpose(torch::kCUDA); }
TEST(manager_lifetime_cuda_local_pooling) { skip_without_cuda(); local_pooling(torch::kCUDA); }
TEST(manager_lifetime_cuda_global_pooling) { skip_without_cuda(); global_pooling(torch::kCUDA); }
TEST(manager_lifetime_cuda_broadcast) { skip_without_cuda(); broadcast(torch::kCUDA); }
TEST(manager_lifetime_cuda_pruning) { skip_without_cuda(); pruning(torch::kCUDA); }
TEST(manager_lifetime_cuda_instance_norm) { skip_without_cuda(); instance_norm(torch::kCUDA); }
TEST(manager_lifetime_cuda_interpolation) { skip_without_cuda(); interpolation(torch::kCUDA); }
