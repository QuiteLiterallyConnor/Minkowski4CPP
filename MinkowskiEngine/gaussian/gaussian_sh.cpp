#include "gaussian/gaussian_sh.hpp"

#include <vector>

namespace minkowski {
namespace gaussian {

at::Tensor sh_basis(int degree, const at::Tensor &directions) {
  TORCH_CHECK(directions.dim() == 2 && directions.size(1) == 3,
              "directions must have shape [N, 3]");
  const int64_t count = sh_coefficient_count(degree);
  const auto x = directions.select(1, 0);
  const auto y = directions.select(1, 1);
  const auto z = directions.select(1, 2);

  std::vector<at::Tensor> basis;
  basis.reserve(static_cast<size_t>(count));
  basis.push_back(at::full_like(x, kShC0));
  if (degree > 0) {
    constexpr double c1 = 0.4886025119029199;
    basis.push_back(-c1 * y);
    basis.push_back(c1 * z);
    basis.push_back(-c1 * x);
  }
  if (degree > 1) {
    const auto xx = x * x, yy = y * y, zz = z * z;
    const auto xy = x * y, yz = y * z, xz = x * z;
    basis.push_back(1.0925484305920792 * xy);
    basis.push_back(-1.0925484305920792 * yz);
    basis.push_back(0.31539156525252005 * (2.0 * zz - xx - yy));
    basis.push_back(-1.0925484305920792 * xz);
    basis.push_back(0.5462742152960396 * (xx - yy));
    if (degree > 2) {
      basis.push_back(-0.5900435899266435 * y * (3.0 * xx - yy));
      basis.push_back(2.890611442640554 * xy * z);
      basis.push_back(-0.4570457994644658 * y * (4.0 * zz - xx - yy));
      basis.push_back(0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy));
      basis.push_back(-0.4570457994644658 * x * (4.0 * zz - xx - yy));
      basis.push_back(1.445305721320277 * z * (xx - yy));
      basis.push_back(-0.5900435899266435 * x * (xx - 3.0 * yy));
    }
  }
  return at::stack(basis, 1);
}

at::Tensor eval_sh_color(int degree, const at::Tensor &sh,
                         const at::Tensor &directions) {
  const int64_t count = sh_coefficient_count(degree);
  TORCH_CHECK(sh.dim() == 3 && sh.size(2) == 3 && sh.size(1) >= count,
              "sh must have shape [N, K, 3] with K >= (degree + 1)^2");
  TORCH_CHECK(sh.size(0) == directions.size(0),
              "sh and directions must have the same length");
  const auto basis = sh_basis(degree, directions);
  const auto coefficients = sh.slice(1, 0, count);
  const auto radiance = (basis.unsqueeze(-1) * coefficients).sum(1);
  return (radiance + 0.5).clamp_min(0.0);
}

} // namespace gaussian
} // namespace minkowski
