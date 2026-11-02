// Verifies the SuperSplat/PlayCanvas compressed-PLY decoder (including the
// quantized SH bands) against the uncompressed export of the same scene:
// tests/data/test.ply and tests/data/test_compressed.ply hold the same
// 139,410 splats in different orders. Every sampled compressed splat is
// matched to its nearest uncompressed splat by position, and each decoded
// attribute must agree within its quantization error.

#include "gaussian/gaussian_model.hpp"
#include "gaussian/gaussian_ply.hpp"
#ifdef CHECK
#undef CHECK
#endif
#include "utils/test_framework.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

using namespace minkowski::gaussian;

#ifndef MINK_TEST_DATA_DIR
#define MINK_TEST_DATA_DIR "tests/data"
#endif

namespace {

std::string compressed_data_path(const char *name) {
  const char *env = std::getenv("MINK_TEST_DATA_DIR");
  return (std::filesystem::path(env ? env : MINK_TEST_DATA_DIR) / name)
      .string();
}

struct Quantiles {
  double median, p99, max;
};

Quantiles quantiles(const at::Tensor &errors) {
  const auto e = errors.reshape({-1}).to(at::kDouble);
  return {e.quantile(0.5).item<double>(), e.quantile(0.99).item<double>(),
          e.max().item<double>()};
}

void expect_quantiles(const char *name, const at::Tensor &errors,
                      double median_limit, double p99_limit) {
  const auto q = quantiles(errors);
  std::cout << "         " << name << ": median " << q.median << ", p99 "
            << q.p99 << ", max " << q.max << " (limits " << median_limit
            << ", " << p99_limit << ")\n";
  if (!(q.median <= median_limit && q.p99 <= p99_limit))
    FAIL_AT(name << " decode error too large");
}

} // namespace

TEST(gaussian_compressed_ply_matches_uncompressed_scene) {
  const auto raw_path = compressed_data_path("test.ply");
  const auto packed_path = compressed_data_path("test_compressed.ply");
  if (!std::filesystem::exists(raw_path) ||
      !std::filesystem::exists(packed_path))
    throw mink_test::Skip("test.ply / test_compressed.ply not found");
  const auto raw = load_gaussian_model_ply(raw_path);
  const auto packed = load_gaussian_model_ply(packed_path);
  CHECK_EQ(packed.size(), raw.size());
  CHECK_EQ(packed.sh_degree(), raw.sh_degree());

  // 4000 random compressed splats, nearest uncompressed splat by position.
  const int64_t samples = 4000;
  const auto pick = at::randperm(packed.size(),
                                 at::make_generator<at::CPUGeneratorImpl>(5),
                                 at::kLong)
                        .slice(0, 0, samples);
  const auto query = packed.xyz().index_select(0, pick);
  std::vector<at::Tensor> best_d, best_i;
  for (int64_t s = 0; s < samples; s += 250) {
    const auto d = at::cdist(query.slice(0, s, s + 250), raw.xyz());
    const auto m = d.min(1);
    best_d.push_back(std::get<0>(m));
    best_i.push_back(std::get<1>(m));
  }
  const auto dist = at::cat(best_d);
  const auto match = at::cat(best_i);
  const double extent =
      (std::get<0>(raw.xyz().max(0)) - std::get<0>(raw.xyz().min(0)))
          .max()
          .item<double>();
  std::cout << "         scene extent " << extent << "\n";
  // Positions are 11/10/11-bit within 256-splat chunk bounds.
  expect_quantiles("position distance / extent", dist / extent, 2e-4, 2e-3);
  // Matches are one-to-one in the sample.
  CHECK_EQ(std::get<0>(at::_unique(match)).numel(), samples);

  const auto q_packed = packed.rotations().index_select(0, pick);
  const auto q_raw = raw.rotations().index_select(0, match);
  expect_quantiles("rotation 1 - |q.q'|",
                   1.0 - (q_packed * q_raw).sum(1).abs(), 1e-4, 2e-3);
  expect_quantiles(
      "log-scale",
      (packed.scale_raw().index_select(0, pick) -
       raw.scale_raw().index_select(0, match))
          .abs(),
      2e-2, 6e-2);
  expect_quantiles("opacity",
                   (packed.opacities().index_select(0, pick) -
                    raw.opacities().index_select(0, match))
                       .abs(),
                   4e-3, 1e-2);
  // DC color: 8-bit RGB inside per-chunk [min, max] ranges, which may extend
  // beyond [0, 1] (so the uncompressed colors are compared unclamped).
  const auto rgb_raw = raw.colors().index_select(0, match);
  const auto rgb_packed = packed.colors().index_select(0, pick);
  std::cout << "         dc color outside [0, 1]: uncompressed "
            << ((rgb_raw < 0) | (rgb_raw > 1)).to(at::kDouble).mean().item<double>()
            << ", decoded "
            << ((rgb_packed < 0) | (rgb_packed > 1)).to(at::kDouble).mean().item<double>()
            << "\n";
  expect_quantiles("dc color (vs clamped uncompressed, info)",
                   (rgb_packed - rgb_raw.clamp(0.0, 1.0)).abs(), 1.0, 1.0);
  expect_quantiles("dc color", (rgb_packed - rgb_raw).abs(), 4e-3, 2e-2);
  // SH rest: 8-bit over [-4, 4], step 8/256. The packer truncates and the
  // decoder returns bin centers, so |error| <= 4/256 = 0.0156 (a value of 0
  // decodes to +0.0156: most rest coefficients sit at that bound).
  expect_quantiles("sh rest",
                   (packed.f_rest().index_select(0, pick) -
                    raw.f_rest().index_select(0, match).clamp(-4.0, 4.0))
                       .abs(),
                   0.0160, 0.0165);
}
