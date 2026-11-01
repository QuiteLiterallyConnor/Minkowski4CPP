#ifndef MINKOWSKI_GAUSSIAN_GAUSSIAN3D_HPP
#define MINKOWSKI_GAUSSIAN_GAUSSIAN3D_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace minkowski {
namespace gaussian {

template <typename T> using Vector3 = std::array<T, 3>;
template <typename T> using Quaternion = std::array<T, 4>; // (w, x, y, z)
template <typename T> using Matrix3 = std::array<std::array<T, 3>, 3>;

template <typename T> class Gaussian3D {
  static_assert(std::is_floating_point<T>::value,
                "Gaussian3D requires a floating-point scalar");

public:
  Vector3<T> mean{T(0), T(0), T(0)};
  Quaternion<T> orientation{T(1), T(0), T(0), T(0)};
  Vector3<T> scale{T(1), T(1), T(1)};
  T opacity = T(1);
  Vector3<T> color{T(0), T(0), T(0)};
  std::vector<T> features;
  std::optional<Vector3<T>> velocity;
  std::optional<T> timestamp;
  std::optional<int64_t> semantic_class;
  std::optional<int64_t> object_id;

  static constexpr T kMinimumScale = T(1e-6);

  Quaternion<T> normalized_orientation() const {
    T norm_squared = T(0);
    for (T value : orientation) {
      if (!std::isfinite(value))
        throw std::invalid_argument("Gaussian orientation must be finite");
      norm_squared += value * value;
    }
    if (!(norm_squared > T(0)) || !std::isfinite(norm_squared))
      throw std::invalid_argument("Gaussian orientation must have nonzero norm");
    const T inverse_norm = T(1) / std::sqrt(norm_squared);
    Quaternion<T> normalized{};
    for (size_t i = 0; i < normalized.size(); ++i)
      normalized[i] = orientation[i] * inverse_norm;
    return normalized;
  }

  void validate(T minimum_scale = kMinimumScale) const {
    if (!(minimum_scale > T(0)) || !std::isfinite(minimum_scale))
      throw std::invalid_argument("minimum scale must be positive and finite");
    for (T value : mean)
      if (!std::isfinite(value))
        throw std::invalid_argument("Gaussian mean must be finite");
    for (T value : scale)
      if (!std::isfinite(value) || value < minimum_scale)
        throw std::invalid_argument("Gaussian scales must be finite and above minimum_scale");
    for (T value : color)
      if (!std::isfinite(value))
        throw std::invalid_argument("Gaussian color must be finite");
    for (T value : features)
      if (!std::isfinite(value))
        throw std::invalid_argument("Gaussian features must be finite");
    if (!std::isfinite(opacity) || opacity < T(0) || opacity > T(1))
      throw std::invalid_argument("Gaussian opacity must be finite and in [0, 1]");
    if (velocity)
      for (T value : *velocity)
        if (!std::isfinite(value))
          throw std::invalid_argument("Gaussian velocity must be finite");
    if (timestamp && !std::isfinite(*timestamp))
      throw std::invalid_argument("Gaussian timestamp must be finite");
    (void)normalized_orientation();
  }

  Matrix3<T> rotation_matrix() const {
    const auto q = normalized_orientation();
    const T w = q[0], x = q[1], y = q[2], z = q[3];
    return {{{T(1) - T(2) * (y * y + z * z), T(2) * (x * y - z * w),
              T(2) * (x * z + y * w)},
             {T(2) * (x * y + z * w), T(1) - T(2) * (x * x + z * z),
              T(2) * (y * z - x * w)},
             {T(2) * (x * z - y * w), T(2) * (y * z + x * w),
              T(1) - T(2) * (x * x + y * y)}}};
  }

  Matrix3<T> covariance() const {
    validate();
    const auto rotation = rotation_matrix();
    Matrix3<T> result{};
    for (size_t row = 0; row < 3; ++row)
      for (size_t column = 0; column < 3; ++column)
        for (size_t axis = 0; axis < 3; ++axis)
          result[row][column] += rotation[row][axis] * scale[axis] * scale[axis] *
                                 rotation[column][axis];
    return result;
  }

  T determinant() const {
    validate();
    return scale[0] * scale[0] * scale[1] * scale[1] * scale[2] * scale[2];
  }

  Matrix3<T> inverse_covariance() const {
    validate();
    const auto rotation = rotation_matrix();
    Matrix3<T> result{};
    for (size_t row = 0; row < 3; ++row)
      for (size_t column = 0; column < 3; ++column)
        for (size_t axis = 0; axis < 3; ++axis)
          result[row][column] += rotation[row][axis] /
                                (scale[axis] * scale[axis]) *
                                rotation[column][axis];
    return result;
  }

  // Evaluated in the Gaussian's frame, sum_k ((R^T d)_k / s_k)^2, instead of
  // d^T Sigma^-1 d: the explicit inverse has entries ~1/s_min^2 whose float
  // rounding swamps the result once the scales differ by ~1e3 (a float
  // Gaussian with scales {2, 0.5, 1e-3} gave 2.75 instead of 2.89).
  T mahalanobis_squared(const Vector3<T> &point) const {
    validate();
    const auto rotation = rotation_matrix();
    Vector3<T> delta{};
    for (size_t i = 0; i < 3; ++i) {
      if (!std::isfinite(point[i]))
        throw std::invalid_argument("evaluation point must be finite");
      delta[i] = point[i] - mean[i];
    }
    T distance = T(0);
    for (size_t axis = 0; axis < 3; ++axis) {
      T local = T(0);
      for (size_t i = 0; i < 3; ++i)
        local += rotation[i][axis] * delta[i];
      local /= scale[axis];
      distance += local * local;
    }
    return distance;
  }

  T density(const Vector3<T> &point) const {
    const T two_pi = T(6.283185307179586476925286766559);
    const T normalization = T(1) / (std::pow(two_pi, T(1.5)) *
                                    scale[0] * scale[1] * scale[2]);
    return normalization * std::exp(T(-0.5) * mahalanobis_squared(point));
  }

  T overlap_integral(const Gaussian3D<T> &other) const {
    validate();
    other.validate();
    const auto a = covariance();
    const auto b = other.covariance();
    Matrix3<T> sum{};
    for (size_t i = 0; i < 3; ++i)
      for (size_t j = 0; j < 3; ++j)
        sum[i][j] = a[i][j] + b[i][j];

    const T det =
        sum[0][0] * (sum[1][1] * sum[2][2] - sum[1][2] * sum[2][1]) -
        sum[0][1] * (sum[1][0] * sum[2][2] - sum[1][2] * sum[2][0]) +
        sum[0][2] * (sum[1][0] * sum[2][1] - sum[1][1] * sum[2][0]);
    if (!(det > T(0)) || !std::isfinite(det))
      throw std::runtime_error("combined covariance is not positive definite");

    Matrix3<T> inverse{};
    inverse[0][0] = (sum[1][1] * sum[2][2] - sum[1][2] * sum[2][1]) / det;
    inverse[0][1] = (sum[0][2] * sum[2][1] - sum[0][1] * sum[2][2]) / det;
    inverse[0][2] = (sum[0][1] * sum[1][2] - sum[0][2] * sum[1][1]) / det;
    inverse[1][0] = (sum[1][2] * sum[2][0] - sum[1][0] * sum[2][2]) / det;
    inverse[1][1] = (sum[0][0] * sum[2][2] - sum[0][2] * sum[2][0]) / det;
    inverse[1][2] = (sum[0][2] * sum[1][0] - sum[0][0] * sum[1][2]) / det;
    inverse[2][0] = (sum[1][0] * sum[2][1] - sum[1][1] * sum[2][0]) / det;
    inverse[2][1] = (sum[0][1] * sum[2][0] - sum[0][0] * sum[2][1]) / det;
    inverse[2][2] = (sum[0][0] * sum[1][1] - sum[0][1] * sum[1][0]) / det;

    Vector3<T> delta{};
    for (size_t i = 0; i < 3; ++i)
      delta[i] = mean[i] - other.mean[i];
    T quadratic = T(0);
    for (size_t row = 0; row < 3; ++row)
      for (size_t column = 0; column < 3; ++column)
        quadratic += delta[row] * inverse[row][column] * delta[column];
    const T two_pi = T(6.283185307179586476925286766559);
    return std::exp(T(-0.5) * quadratic) /
           (std::pow(two_pi, T(1.5)) * std::sqrt(det));
  }

  Vector3<T> sample(std::mt19937_64 &generator) const {
    validate();
    std::normal_distribution<T> normal(T(0), T(1));
    Vector3<T> local{scale[0] * normal(generator), scale[1] * normal(generator),
                     scale[2] * normal(generator)};
    const auto rotation = rotation_matrix();
    Vector3<T> result = mean;
    for (size_t row = 0; row < 3; ++row)
      for (size_t column = 0; column < 3; ++column)
        result[row] += rotation[row][column] * local[column];
    return result;
  }

  std::pair<Vector3<T>, Vector3<T>> bounds(T confidence_radius = T(3)) const {
    validate();
    if (!(confidence_radius > T(0)) || !std::isfinite(confidence_radius))
      throw std::invalid_argument("confidence radius must be positive and finite");
    const auto cov = covariance();
    Vector3<T> lower{}, upper{};
    for (size_t axis = 0; axis < 3; ++axis) {
      const T extent = confidence_radius * std::sqrt(cov[axis][axis]);
      lower[axis] = mean[axis] - extent;
      upper[axis] = mean[axis] + extent;
    }
    return {lower, upper};
  }
};

using Gaussian3Df = Gaussian3D<float>;
using Gaussian3Dd = Gaussian3D<double>;

} // namespace gaussian
} // namespace minkowski

#endif
