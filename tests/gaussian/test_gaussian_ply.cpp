#include "gaussian/gaussian_ply.hpp"
#include "gaussian/gaussian_optimizer.hpp"
#ifdef CHECK
#undef CHECK
#endif
#include "utils/test_framework.hpp"

#include <ATen/ATen.h>
#ifdef USE_CUDA
#include <ATen/cuda/CUDAContext.h>
#endif
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using minkowski::gaussian::GaussianOptimizer;
using minkowski::gaussian::GaussianTensor;
using minkowski::gaussian::load_gaussian_ply;

#ifndef MINK_TEST_DATA_DIR
#define MINK_TEST_DATA_DIR "tests/data"
#endif

namespace {

std::string ply_path(const char *name) {
  const char *env = std::getenv("MINK_TEST_DATA_DIR");
  return (std::filesystem::path(env ? env : MINK_TEST_DATA_DIR) / name)
      .string();
}

// Loaded once and shared; tests only use const or cloned copies.
const GaussianTensor &load_scene(const char *name) {
  const std::string path = ply_path(name);
  if (!std::filesystem::exists(path))
    throw mink_test::Skip(std::string(name) + " not found at " + path);
  return *new GaussianTensor(load_gaussian_ply(path));
}

const GaussianTensor &scene() {
  static const GaussianTensor &loaded = load_scene("test.ply");
  return loaded;
}

const GaussianTensor &compressed_scene() {
  static const GaussianTensor &loaded = load_scene("test_compressed.ply");
  return loaded;
}

} // namespace

TEST(ply_compressed_and_uncompressed_are_both_detected) {
  const auto &raw = scene();
  const auto &packed = compressed_scene();
  CHECK(packed.dtype() == at::kFloat);
  CHECK(at::isfinite(packed.positions()).all().item<bool>());
  CHECK((packed.scales() > 0).all().item<bool>());
  CHECK((packed.opacities() >= 0).all().item<bool>());
  CHECK((packed.opacities() <= 1).all().item<bool>());
  const auto norms = packed.rotations().norm(2, {1});
  CHECK_ALLCLOSE(norms, at::ones_like(norms), 1e-3, 1e-3);
  CHECK(raw.size() > 0 && packed.size() > 0);
}

TEST(ply_loads_expected_layout) {
  const auto &g = scene();
  CHECK(g.size() > 0);
  CHECK_EQ(g.positions().sizes().vec(), (std::vector<int64_t>{g.size(), 3}));
  CHECK_EQ(g.rotations().size(1), 4);
  CHECK_EQ(g.scales().size(1), 3);
  CHECK_EQ(g.opacities().size(1), 1);
  CHECK_EQ(g.colors().size(1), 3);
  CHECK(g.dtype() == at::kFloat);
  CHECK(g.device().is_cpu());
}

TEST(ply_values_are_valid) {
  const auto &g = scene();
  CHECK(at::isfinite(g.positions()).all().item<bool>());
  CHECK(at::isfinite(g.scales()).all().item<bool>());
  CHECK(at::isfinite(g.colors()).all().item<bool>());
  CHECK((g.scales() > 0).all().item<bool>());
  CHECK((g.opacities() >= 0).all().item<bool>());
  CHECK((g.opacities() <= 1).all().item<bool>());
  const auto norms = g.rotations().norm(2, {1});
  CHECK_ALLCLOSE(norms, at::ones_like(norms), 1e-4, 1e-4);
}

TEST(ply_covariances_are_symmetric_positive_semidefinite) {
  const auto &g = scene();
  const auto cov = g.covariances();
  CHECK_EQ(cov.sizes().vec(), (std::vector<int64_t>{g.size(), 3, 3}));
  CHECK_ALLCLOSE(cov, cov.transpose(1, 2), 1e-3, 1e-6);
  // Sample to keep the eigen-decomposition cheap.
  const auto sample = cov.slice(0, 0, std::min<int64_t>(g.size(), 2000))
                          .to(at::kDouble);
  const auto eig = at::linalg_eigvalsh(sample);
  CHECK((eig >= -1e-9).all().item<bool>());
}

TEST(ply_filter_opacity_and_bounds) {
  const auto &g = scene();
  const auto visible = g.filter_opacity(0.5);
  CHECK(visible.size() <= g.size());
  CHECK((visible.opacities() >= 0.5).all().item<bool>());
  CHECK_EQ(visible.size(), (g.opacities() >= 0.5).sum().item<int64_t>());

  const auto lo = std::get<0>(g.positions().min(0));
  const auto hi = std::get<0>(g.positions().max(0));
  const auto mid = 0.5 * (lo + hi);
  const auto half = g.filter_bounds(lo, mid);
  CHECK(half.size() > 0);
  CHECK(half.size() < g.size());
  CHECK((half.positions() <= mid + 1e-6).all().item<bool>());
  CHECK_EQ(g.filter_bounds(lo - 1e-3, hi + 1e-3).size(), g.size());
}

TEST(ply_select_remove_partition) {
  const auto &g = scene();
  const auto idx = at::arange(0, g.size(), 2, at::kLong);
  const auto kept = g.select(idx);
  const auto rest = g.remove(idx);
  CHECK_EQ(kept.size() + rest.size(), g.size());
  const auto merged = kept.append(rest);
  CHECK_EQ(merged.size(), g.size());
}

TEST(ply_transform_round_trip_restores_scene) {
  const auto &g = scene();
  const double angle = 0.7;
  const auto q = at::tensor({std::cos(angle / 2), 0.0, 0.0, std::sin(angle / 2)},
                            at::kFloat)
                     .reshape({4});
  const auto t = at::tensor({1.5, -2.0, 0.25}, at::kFloat);
  const auto moved = g.transform(q, t);
  CHECK(!at::allclose(moved.positions(), g.positions(), 1e-3, 1e-3));

  // Inverse: rotate by conjugate, translate by -R^T t.
  const auto qc = at::tensor({q[0].item<float>(), 0.0f, 0.0f,
                              -q[3].item<float>()},
                             at::kFloat);
  const auto back = moved.transform(qc, at::zeros({3}, at::kFloat));
  const auto c = std::cos(angle), s = std::sin(angle);
  const auto rt_t = at::tensor({c * 1.5 + s * -2.0, -s * 1.5 + c * -2.0, 0.25},
                               at::kFloat);
  const auto restored = back.transform(at::tensor({1.f, 0.f, 0.f, 0.f}),
                                       -rt_t);
  CHECK_ALLCLOSE(restored.positions(), g.positions(), 1e-3, 1e-3);
  CHECK_ALLCLOSE(restored.scales(), g.scales(), 1e-6, 1e-6);
}

TEST(ply_optimizer_reduces_color_loss) {
  const auto &g = scene();
  auto work = g.select(at::arange(0, std::min<int64_t>(g.size(), 5000), at::kLong))
                  .clone();
  const auto target = at::zeros_like(work.colors());
  GaussianOptimizer optimizer(work, 1e-2);
  double first = 0.0, last = 0.0;
  for (int i = 0; i < 10; ++i) {
    const auto loss = (work.colors() - target).pow(2).mean();
    if (i == 0)
      first = loss.item<double>();
    last = loss.item<double>();
    optimizer.step(loss);
  }
  CHECK(last < first);
}

TEST(ply_rejects_bad_files) {
  const auto dir = std::filesystem::temp_directory_path();
  CHECK_THROWS(load_gaussian_ply((dir / "mink_missing_file.ply").string()));
  const auto bad = dir / "mink_bad_header.ply";
  {
    std::ofstream out(bad, std::ios::binary);
    out << "not a ply\n";
  }
  CHECK_THROWS(load_gaussian_ply(bad.string()));
  const auto truncated = dir / "mink_truncated.ply";
  {
    std::ofstream out(truncated, std::ios::binary);
    out << "ply\nformat binary_little_endian 1.0\nelement vertex 10\n"
           "property float x\nend_header\n";
  }
  CHECK_THROWS(load_gaussian_ply(truncated.string()));
  std::filesystem::remove(bad);
  std::filesystem::remove(truncated);
}

#ifdef USE_CUDA
TEST(ply_cuda_transform_matches_cpu) {
  if (!at::cuda::is_available())
    throw mink_test::Skip("CUDA runtime or device is unavailable");
  const auto &g = scene();
  const auto q = at::tensor({0.9238795f, 0.f, 0.3826834f, 0.f});
  const auto t = at::tensor({0.5f, 1.0f, -1.0f});
  const auto cpu = g.transform(q, t);
  const auto gpu = g.to(at::kCUDA).transform(q.cuda(), t.cuda()).to(at::kCPU);
  CHECK_ALLCLOSE(gpu.positions(), cpu.positions(), 1e-4, 1e-4);
  CHECK_ALLCLOSE(gpu.rotations(), cpu.rotations(), 1e-4, 1e-4);
  CHECK_ALLCLOSE(g.to(at::kCUDA).covariances().cpu(), g.covariances(), 1e-3,
                 1e-6);
}
#endif
