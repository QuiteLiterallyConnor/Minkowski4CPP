#include "gaussian/gaussian3d.hpp"
#include "utils/test_framework.hpp"

#include <cmath>
#include <random>

using minkowski::gaussian::Gaussian3Dd;

TEST(gaussian3d_identity_covariance_and_density) {
  Gaussian3Dd gaussian;
  const auto covariance = gaussian.covariance();
  for (int row = 0; row < 3; ++row)
    for (int column = 0; column < 3; ++column)
      CHECK_NEAR(covariance[row][column], row == column ? 1.0 : 0.0, 1e-14);
  CHECK_NEAR(gaussian.determinant(), 1.0, 1e-14);
  CHECK_NEAR(gaussian.density({0.0, 0.0, 0.0}),
             1.0 / std::pow(2.0 * 3.14159265358979323846, 1.5), 1e-14);
  CHECK_NEAR(gaussian.mahalanobis_squared({1.0, 2.0, 2.0}), 9.0, 1e-14);
}

TEST(gaussian3d_anisotropic_rotated_covariance_is_symmetric) {
  Gaussian3Dd gaussian;
  gaussian.scale = {2.0, 1.0, 0.5};
  const double half_angle = 3.14159265358979323846 / 8.0;
  gaussian.orientation = {std::cos(half_angle), 0.0, 0.0,
                          std::sin(half_angle)};
  const auto covariance = gaussian.covariance();
  for (int row = 0; row < 3; ++row)
    for (int column = 0; column < 3; ++column)
      CHECK_NEAR(covariance[row][column], covariance[column][row], 1e-14);
  CHECK_NEAR(gaussian.determinant(), 1.0, 1e-14);
}

TEST(gaussian3d_rejects_invalid_parameters) {
  Gaussian3Dd gaussian;
  gaussian.scale[1] = 0.0;
  CHECK_THROWS(gaussian.validate());
  gaussian.scale = {1.0, 1.0, 1.0};
  gaussian.orientation = {0.0, 0.0, 0.0, 0.0};
  CHECK_THROWS(gaussian.covariance());
  gaussian.orientation = {1.0, 0.0, 0.0, 0.0};
  gaussian.opacity = 1.1;
  CHECK_THROWS(gaussian.validate());
}

TEST(gaussian3d_overlap_sampling_and_bounds) {
  Gaussian3Dd gaussian;
  gaussian.scale = {2.0, 1.0, 0.5};
  CHECK(gaussian.overlap_integral(gaussian) > 0.0);
  const auto [lower, upper] = gaussian.bounds();
  for (int axis = 0; axis < 3; ++axis) {
    CHECK(lower[axis] < gaussian.mean[axis]);
    CHECK(upper[axis] > gaussian.mean[axis]);
  }
  std::mt19937_64 rng(7);
  const auto sample = gaussian.sample(rng);
  for (double coordinate : sample)
    CHECK(std::isfinite(coordinate));
}
