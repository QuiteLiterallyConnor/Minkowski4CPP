#include <iostream>
#include <vector>
#include "gaussian/gaussian_camera.hpp"
#include "gaussian/gaussian_cuda.hpp"
#include "gaussian/gaussian_render.hpp"
#include "utils/test_framework.hpp"

#ifdef USE_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

using namespace minkowski::gaussian;
namespace gd = minkowski::gaussian::detail;
namespace {
void require_cuda() {
  if (!torch::cuda::is_available())
    throw mink_test::Skip("CUDA not available");
}
at::Tensor reference(const std::vector<at::Tensor> &v, const at::Tensor &ids) {
  auto take = [&](const at::Tensor &t) { return t.index_select(0, ids); };
  return at::cat({take(v[0]), take(v[1].slice(1, 2, 5)),
                  take(v[2].reshape({-1, 1})), take(v[1].slice(1, 5, 6)), take(v[3])}, 1);
}
at::Tensor fused(const std::vector<at::Tensor> &v, const at::Tensor &ids) {
  return gd::gather_attributes(v[0], v[1], v[2], v[3], ids);
}
struct GatherToggle {
  bool saved = gd::cuda_fused_gather();
  ~GatherToggle() { gd::cuda_fused_gather() = saved; }
};
} // namespace

TEST(gaussian_fused_gather_exact_permutation_and_gradient) {
  require_cuda();
  torch::manual_seed(71);
  const auto f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  for (const int64_t n : {0, 1, 7, 1025}) {
    for (const int64_t m : {int64_t(0), n / 2, n}) {
      // Deliberately noncontiguous inputs, including both means strides and [N,1] opacity.
      std::vector<at::Tensor> v{torch::randn({4, n}, f).transpose(0, 1).slice(1, 0, 4, 2),
                                torch::randn({n, 20}, f).slice(1, 0, 20, 2),
                                torch::randn({n, 2}, f).slice(1, 0, 1),
                                torch::randn({3, n}, f).transpose(0, 1)};
      for (auto &t : v) t.requires_grad_(true);
      const auto ids = torch::randperm(n, f.dtype(torch::kLong)).slice(0, 0, m);
      const auto a = fused(v, ids), b = reference(v, ids);
      CHECK(a.cpu().equal(b.cpu()));
      // Noncontiguous upstream gradient must also be supported.
      const auto weight = torch::randn({10, m}, f).transpose(0, 1);
      const auto ga = torch::autograd::grad({a}, v, {weight});
      const auto gb = torch::autograd::grad({b}, v, {weight});
      for (size_t j = 0; j < v.size(); ++j) {
        CHECK(ga[j].sizes() == v[j].sizes());
        CHECK(ga[j].cpu().equal(gb[j].cpu()));
      }
    }
  }
}

TEST(gaussian_fused_gather_finite_difference) {
  require_cuda();
  const auto f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  // Binary fractions make this linear finite-difference check exact in float32.
  std::vector<at::Tensor> v{torch::zeros({7, 2}, f), torch::zeros({7, 10}, f),
                            torch::zeros({7}, f), torch::zeros({7, 3}, f)};
  for (auto &t : v) t.requires_grad_(true);
  const auto ids = torch::tensor({6, 1, 3}, torch::kLong).to(torch::kCUDA);
  const auto weight = (torch::arange(30, f) / 32).reshape({3, 10});
  const auto grads = torch::autograd::grad({fused(v, ids)}, v, {weight});
  torch::NoGradGuard ng;
  constexpr double eps = 1.0 / 128;
  for (size_t j = 0; j < v.size(); ++j) {
    auto flat = v[j].view({-1});
    for (int64_t i = 0; i < flat.numel(); ++i) {
      flat[i] = eps;
      const double plus = (fused(v, ids) * weight).sum().item<double>();
      flat[i] = -eps;
      const double minus = (fused(v, ids) * weight).sum().item<double>();
      flat[i] = 0;
      CHECK_NEAR((plus - minus) / (2 * eps), grads[j].reshape({-1})[i].item<double>(), 0.0);
    }
  }
}

TEST(gaussian_fused_gather_current_stream_and_temporaries) {
  require_cuda();
  const auto stream = c10::cuda::getStreamFromPool();
  const c10::cuda::CUDAStreamGuard guard(stream);
  const auto f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  const int64_t n = 2049;
  std::vector<at::Tensor> v{
      torch::arange(n * 4, f).view({4, n}).transpose(0, 1).slice(1, 0, 4, 2),
      torch::arange(n * 20, f).view({n, 20}).slice(1, 0, 20, 2),
      torch::arange(n * 2, f).view({n, 2}).slice(1, 0, 1),
      torch::arange(n * 3, f).view({3, n}).transpose(0, 1)};
  for (auto &t : v) t.requires_grad_(true);
  const auto ids = torch::arange(n, f.dtype(torch::kLong)).flip({0}).slice(0, 0, n - 7);
  const auto a = fused(v, ids);
  // The contiguous input temporaries have been released, but their queued reads
  // must precede allocator reuse on this nondefault stream.
  for (int i = 0; i < 4; ++i) {
    auto scratch = torch::empty({n, 20}, f);
    scratch.fill_(i + 1);
  }
  const auto b = reference(v, ids);
  const auto weight = torch::arange(10 * ids.numel(), f).view({10, ids.numel()}).transpose(0, 1);
  const auto ga = torch::autograd::grad({a}, v, {weight});
  const auto gb = torch::autograd::grad({b}, v, {weight});
  stream.synchronize();
  CHECK(a.cpu().equal(b.cpu()));
  for (size_t j = 0; j < v.size(); ++j) CHECK(ga[j].cpu().equal(gb[j].cpu()));
}

TEST(gaussian_fused_gather_rejects_invalid_metadata) {
  require_cuda();
  const auto f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  const auto m = torch::zeros({2, 2}, f), p = torch::zeros({2, 10}, f);
  const auto o = torch::zeros({2, 1}, f), c = torch::zeros({2, 3}, f);
  const auto ids = torch::arange(2, f.dtype(torch::kLong));
  CHECK_THROWS(gd::gather_attributes(m, p, o, c, ids.to(torch::kInt)));
  CHECK_THROWS(gd::gather_attributes(m, p, o, c, ids.cpu()));
  CHECK_THROWS(gd::gather_attributes(m, p, o, c, ids.reshape({1, 2})));
  CHECK_THROWS(gd::gather_attributes(m, p, o, c.slice(1, 0, 2), ids));
  CHECK_THROWS(gd::gather_attributes(m.to(torch::kDouble), p, o, c, ids));
  // Index values are the internal caller's contract, not an unchecked public API.
}

TEST(gaussian_fused_gather_frozen_inputs_do_not_retain_unused_gradients) {
  require_cuda();
  const auto f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  const int64_t n = 257;
  const auto ids = torch::arange(n, f.dtype(torch::kLong)).flip({0}).slice(0, 0, n - 3);
  for (size_t trainable = 0; trainable < 4; ++trainable) {
    std::vector<at::Tensor> v{torch::zeros({n, 2}, f), torch::zeros({n, 10}, f),
                              torch::zeros({n, 1}, f), torch::zeros({n, 3}, f)};
    v[trainable].requires_grad_(true);
    const auto weight = torch::arange(ids.numel() * 10, f).view({ids.numel(), 10});
    const auto got = torch::autograd::grad({fused(v, ids)}, {v[trainable]}, {weight})[0];
    const auto expected = torch::autograd::grad({reference(v, ids)}, {v[trainable]}, {weight})[0];
    CHECK(got.cpu().equal(expected.cpu()));
    std::cout << "         trainable attribute " << trainable << ": gradient storage "
              << got.storage().nbytes() << " vs " << expected.storage().nbytes() << " bytes\n";
    // A retained color/opacity gradient must not keep buffers for frozen geometry alive.
    CHECK(got.storage().nbytes() <= static_cast<size_t>(got.numel() * got.element_size()));
  }
}

TEST(gaussian_fused_gather_render_retains_means_gradient) {
  require_cuda();
  GatherToggle restore;
  torch::manual_seed(73);
  const auto f = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  const auto cam = Camera::look_at({0.3, -0.2, -3.0}, {0, 0, 0}, {0, -1, 0}, 131, 77, 1.0);
  const int64_t n = 3000;
  std::vector<at::Tensor> base{torch::randn({n, 3}, f) * 1.2,
                              torch::randn({n, 4}, f),
                              torch::rand({n, 3}, f) * 0.15 + 0.005,
                              torch::rand({n, 1}, f) * 0.9 + 0.05,
                              torch::rand({n, 3}, f)};
  base[0].slice(0, 0, 100).select(1, 2).fill_(-10); // culled rows
  base[2].slice(0, 200, 400).select(1, 1).fill_(1e-5); // needle covariances
  std::vector<RenderOutput> outs;
  std::vector<std::vector<at::Tensor>> gradients;
  for (bool use_fused : {false, true}) {
    gd::cuda_fused_gather() = use_fused;
    std::vector<at::Tensor> v;
    for (auto &t : base) v.push_back(t.detach().clone().requires_grad_(true));
    auto out = render_gaussians(v[0], v[1], v[2], v[3], v[4], cam);
    (out.image.square().sum() + 0.25 * out.depth.sum() + out.alpha.sum()).backward();
    std::vector<at::Tensor> g;
    for (auto &t : v) g.push_back(t.grad());
    g.push_back(out.means2d.grad());
    outs.push_back(out);
    gradients.push_back(g);
  }
  CHECK(outs[0].image.cpu().equal(outs[1].image.cpu()));
  CHECK(outs[0].depth.cpu().equal(outs[1].depth.cpu()));
  CHECK(outs[0].alpha.cpu().equal(outs[1].alpha.cpu()));
  CHECK(outs[0].visible.cpu().equal(outs[1].visible.cpu()));
  for (size_t j = 0; j < gradients[0].size(); ++j) {
    CHECK(gradients[0][j].defined() && gradients[1][j].defined());
    const double err = (gradients[0][j] - gradients[1][j]).abs().max().item<double>();
    const double scale = gradients[0][j].abs().max().item<double>();
    std::cout << "         gather render grad " << j << " relative diff " << err / scale << "\n";
    CHECK(err <= 1e-5 * scale + 1e-6); // rasterizer atomic accumulation ordering
  }
}

#endif
