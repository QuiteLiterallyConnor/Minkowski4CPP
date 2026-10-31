#ifndef MINKOWSKI_GAUSSIAN_GAUSSIAN_SH_HPP
#define MINKOWSKI_GAUSSIAN_GAUSSIAN_SH_HPP

#include <torch/torch.h>

namespace minkowski {
namespace gaussian {

constexpr double kShC0 = 0.28209479177387814;

// Number of spherical-harmonic coefficients per channel for a degree 0..3.
inline int64_t sh_coefficient_count(int degree) {
  TORCH_CHECK(degree >= 0 && degree <= 3, "SH degree must be in [0, 3]");
  return static_cast<int64_t>((degree + 1) * (degree + 1));
}

// Real SH basis for unit directions [N, 3]; returns [N, (degree + 1)^2].
at::Tensor sh_basis(int degree, const at::Tensor &directions);

// Evaluates view-dependent color: clamp_min(0.5 + sum_k basis_k * sh_k, 0).
// `sh` is [N, K, 3] with K >= (degree + 1)^2 (index 0 is the DC term) and
// `directions` are unit vectors [N, 3] from the camera to each Gaussian.
at::Tensor eval_sh_color(int degree, const at::Tensor &sh,
                         const at::Tensor &directions);

} // namespace gaussian
} // namespace minkowski

#endif
