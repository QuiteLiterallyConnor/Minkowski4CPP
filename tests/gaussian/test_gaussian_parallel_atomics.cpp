#include "gaussian/gaussian_cuda.hpp"
#include "utils/test_framework.hpp"
#include <ATen/ATen.h>
#ifdef USE_CUDA
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif
#include <algorithm>
#include <iostream>

#ifdef USE_CUDA
namespace gd = minkowski::gaussian::detail;
namespace {
struct Toggle {
  bool saved = gd::cuda_raster_backward_parallel_atomics();
  ~Toggle() { gd::cuda_raster_backward_parallel_atomics() = saved; }
};
void require_cuda() {
  if (!at::cuda::is_available()) throw mink_test::Skip("CUDA not available");
}
struct Fixture {
  int height = 19, width = 35;
  at::Tensor attrs, ids, ranges, bg, out, trans, last, weight;
  explicit Fixture(int per_tile) {
    const int tiles = ((height + 15) / 16) * ((width + 15) / 16);
    const int n = per_tile * tiles;
    auto cpu = at::zeros({n, 10}, at::kFloat);
    auto *a = cpu.data_ptr<float>();
    auto r = at::zeros({tiles, 2}, at::kInt);
    for (int tile = 0; tile < tiles; ++tile) {
      r[tile][0] = tile * per_tile;
      r[tile][1] = (tile + 1) * per_tile;
      for (int j = 0; j < per_tile; ++j) {
        float *v = a + static_cast<int64_t>(tile * per_tile + j) * 10;
        v[0] = (tile % 3) * 16 + 8.f;
        v[1] = (tile / 3) * 16 + 1.f + 2.f * (8 * j / std::max(1, per_tile));
        v[2] = 0.02f; v[4] = 4.f; v[5] = 0.8f;
        v[6] = 1.f + j * 0.001f;
        v[7] = 0.1f + (j % 11) * 0.05f;
        v[8] = 0.2f + (j % 7) * 0.04f;
        v[9] = 0.3f + (j % 5) * 0.02f;
      }
    }
    const auto opts = at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
    attrs = cpu.to(at::kCUDA);
    ids = at::arange(n, at::kInt).to(at::kCUDA);
    ranges = r.to(at::kCUDA);
    bg = at::tensor({0.3f, 0.2f, 0.1f}).to(at::kCUDA);
    out = at::empty({height, width, 5}, opts);
    trans = at::empty({height, width}, opts);
    last = at::empty({height, width}, opts.dtype(at::kInt));
    // Use a nonuniform loss, including depth and alpha, so all ten gradients matter.
    weight = (at::arange(height * width * 5, opts).remainder(29) / 29 - 0.4).view({height, width, 5});
  }
  void forward() {
    gd::rasterize_forward_cuda(attrs, ids, ranges, bg, height, width, out, trans, last);
  }
  at::Tensor backward(bool bounded) {
    gd::cuda_raster_backward_parallel_atomics() = bounded;
    auto grad = at::zeros_like(attrs);
    gd::rasterize_backward_cuda(attrs, ids, ranges, bg, weight, trans, last, height, width, grad);
    return grad;
  }
};
} // namespace
#endif

TEST(gaussian_parallel_atomics_match_serial_atomics_at_batch_edges) {
#ifdef USE_CUDA
  require_cuda(); Toggle restore;
  for (const int per_tile : {0, 1, 31, 255, 256, 257, 513}) {
    Fixture f(per_tile);
    f.forward();
    const auto old = f.backward(false).cpu(), bounded = f.backward(true).cpu();
    CHECK(old.sizes() == bounded.sizes());
    if (per_tile) {
      const double scale = old.abs().max().item<double>();
      const double diff = (old - bounded).abs().max().item<double>();
      std::cout << "         per_tile=" << per_tile << " max grad diff=" << diff << '\n';
      CHECK(scale > 0);
      CHECK(diff <= 2e-5 * scale + 2e-6); // existing float atomic accumulation order
      CHECK(at::isfinite(bounded).all().item<bool>());
    }
  }
#else
  throw mink_test::Skip("built without CUDA");
#endif
}

TEST(gaussian_parallel_atomics_nondefault_stream_and_zero_loss) {
#ifdef USE_CUDA
  require_cuda(); Toggle restore;
  const auto stream = c10::cuda::getStreamFromPool();
  const c10::cuda::CUDAStreamGuard guard(stream);
  Fixture f(513);
  f.forward();
  const auto old = f.backward(false), bounded = f.backward(true);
  stream.synchronize();
  CHECK(at::allclose(old, bounded, 2e-5, 2e-6));
  f.weight.zero_();
  CHECK(f.backward(true).abs().sum().item<double>() == 0.0);
#else
  throw mink_test::Skip("built without CUDA");
#endif
}

TEST(gaussian_parallel_atomics_finite_difference) {
#ifdef USE_CUDA
  require_cuda(); Toggle restore;
  Fixture f(1);
  // Keep away from opacity saturation and early termination. Only inspect the
  // first tile's Gaussian; neighboring lists intentionally contain different rows.
  auto cpu = f.attrs.cpu();
  cpu.select(1, 4).fill_(0.04f);
  cpu.select(1, 5).fill_(0.15f);
  cpu.select(1, 1).add_(5.3f);
  f.attrs = cpu.to(at::kCUDA);
  f.forward();
  const auto grad = f.backward(true).cpu();
  const auto weight = f.weight.cpu().to(at::kDouble);
  auto loss = [&]() {
    f.attrs = cpu.to(at::kCUDA);
    f.forward();
    return (f.out.cpu().to(at::kDouble) * weight).sum().item<double>();
  };
  for (int k = 0; k < 10; ++k) {
    const float original = cpu[0][k].item<float>();
    const float eps = (k >= 2 && k <= 4) ? 1e-4f : 1e-3f;
    cpu[0][k] = original + eps; const double plus = loss();
    cpu[0][k] = original - eps; const double minus = loss();
    cpu[0][k] = original;
    const double numerical = (plus - minus) / (2 * eps);
    const double analytical = grad[0][k].item<double>();
    std::cout << "         attr=" << k << " FD=" << numerical << " analytic=" << analytical << '\n';
    CHECK(std::abs(numerical - analytical) <= 0.01 * std::abs(analytical) + 0.002);
  }
#else
  throw mink_test::Skip("built without CUDA");
#endif
}
