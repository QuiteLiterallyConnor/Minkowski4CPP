// The autograd graph must keep the coordinate manager alive: backward() has to work after every
// SparseTensor that referred to the manager is gone, e.g.
//   conv->forward(SparseTensor(feats, coords)).F().sum().backward();
// and the manager must be released together with the graph (no leak).
#include <memory>

#include "core_test_utils.hpp"

using namespace minkowski;

namespace {

// `op` maps a sparse tensor to a dense feature tensor through one autograd function. It runs on a
// tensor whose manager is only referenced by SparseTensors local to this helper; afterwards the
// gradient must match a run where the manager is still alive.
template <typename Op>
void check_graph_keeps_manager(Op op, int64_t channels, uint64_t seed) {
    auto coords = core_test::random_coords(30, 5, 2, 3, seed);
    torch::manual_seed(seed);
    auto feats0 = torch::randn({coords.size(0), channels}, torch::kFloat64);

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
    CHECK_ALLCLOSE(feats.grad(), feats_ref.grad(), 1e-12, 1e-12);

    out = at::Tensor();  // releasing the graph releases the manager
    if (!weak.expired()) FAIL_AT("coordinate manager leaked after the graph was released");
}

}  // namespace

TEST(manager_lifetime_convolution) {
    MinkowskiConvolution conv(2, 3, 3, 2, 1, false, false, ConvolutionMode::DEFAULT, 3);
    conv->to(torch::kFloat64);
    check_graph_keeps_manager([&](const SparseTensor& x) { return conv->forward(x).F(); }, 2, 51);
}

TEST(manager_lifetime_convolution_transpose) {
    MinkowskiConvolutionTranspose conv(2, 3, 3, 1, 1, false, false, ConvolutionMode::DEFAULT, 3);
    conv->to(torch::kFloat64);
    check_graph_keeps_manager([&](const SparseTensor& x) { return conv->forward(x).F(); }, 2, 52);
}

TEST(manager_lifetime_local_pooling) {
    MinkowskiAvgPooling pool(2, 2, 1, 3);
    check_graph_keeps_manager([&](const SparseTensor& x) { return pool->forward(x).F(); }, 3, 53);
}

TEST(manager_lifetime_global_pooling) {
    MinkowskiGlobalMaxPooling pool;
    check_graph_keeps_manager([&](const SparseTensor& x) { return pool->forward(x).F(); }, 3, 54);
}

TEST(manager_lifetime_broadcast) {
    MinkowskiGlobalAvgPooling pool;
    MinkowskiBroadcastMultiplication mul;
    check_graph_keeps_manager(
        [&](const SparseTensor& x) {
            // Detached global features: only the broadcast function sees x's graph.
            auto glob = pool->forward(x);
            SparseTensor g(glob.F().detach(), glob.coordinate_map_key(), glob.coordinate_manager());
            return mul->forward(x, g).F();
        },
        3, 55);
}

TEST(manager_lifetime_pruning) {
    MinkowskiPruning prune;
    check_graph_keeps_manager(
        [&](const SparseTensor& x) {
            auto mask = torch::arange(x.F().size(0)).remainder(3) != 0;
            return prune->forward(x, mask).F();
        },
        2, 56);
}

TEST(manager_lifetime_instance_norm) {
    MinkowskiInstanceNorm norm(3);
    norm->to(torch::kFloat64);
    check_graph_keeps_manager([&](const SparseTensor& x) { return norm->forward(x).F(); }, 3, 57);
}

TEST(manager_lifetime_interpolation) {
    MinkowskiInterpolation interp;
    check_graph_keeps_manager(
        [&](const SparseTensor& x) {
            auto field = x.C().to(torch::kFloat64);  // batch index, then x y z
            field.slice(1, 1).add_(0.25);
            return interp->forward(x, field);
        },
        2, 58);
}
