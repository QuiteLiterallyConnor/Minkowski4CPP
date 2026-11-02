#ifndef MINK_CPP_SPHERE_GEOMETRY_HPP
#define MINK_CPP_SPHERE_GEOMETRY_HPP

// Cell geometry shared by the CPU (sphere.cpp) and CUDA (sphere_cuda.cu) sphere
// voxelizers so both make identical decisions.
#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define MINK_SPHERE_HD __host__ __device__
#else
#define MINK_SPHERE_HD
#endif

namespace minkowski {
namespace adaptive {
namespace sphere_detail {

constexpr double kPi = 3.14159265358979323846;

// Computed in a frame where the sphere center is (r, r, r) and the bounding cube
// [0, 2r)^3 is the level-0 cell; a cell is (x, y, z, level).
struct SphereGeometry {
  double r;

  MINK_SPHERE_HD double side() const { return 2.0 * r; }
  MINK_SPHERE_HD double exact_volume() const { return (4.0 / 3.0) * kPi * r * r * r; }
  MINK_SPHERE_HD double accuracy(double volume) const {
    return 100.0 * (1.0 - fabs(volume - exact_volume()) / exact_volume());
  }
  MINK_SPHERE_HD double cell_size(int level) const { return ldexp(side(), -level); }

  // Intersects the sphere surface (neither fully inside nor fully outside).
  MINK_SPHERE_HD bool is_boundary(int64_t x, int64_t y, int64_t z, int level) const {
    const double size = cell_size(level);
    const int64_t c[3] = {x, y, z};
    double nearest2 = 0.0, farthest2 = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      const double lo = static_cast<double>(c[axis]) * size;
      const double hi = lo + size;
      const double clamped = r < lo ? lo : (r > hi ? hi : r);
      const double nearest = clamped - r;
      const double d_lo = fabs(lo - r), d_hi = fabs(hi - r);
      const double farthest = d_lo > d_hi ? d_lo : d_hi;
      nearest2 += nearest * nearest;
      farthest2 += farthest * farthest;
    }
    return nearest2 < r * r && farthest2 > r * r;
  }

  // A leaf counts as occupied when its center is inside the sphere.
  MINK_SPHERE_HD bool occupied(int64_t x, int64_t y, int64_t z, int level) const {
    const double size = cell_size(level);
    const int64_t c[3] = {x, y, z};
    double d2 = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      const double d = (static_cast<double>(c[axis]) + 0.5) * size - r;
      d2 += d * d;
    }
    return d2 <= r * r;
  }

  MINK_SPHERE_HD double contribution(int64_t x, int64_t y, int64_t z, int level) const {
    const double size = cell_size(level);
    return occupied(x, y, z, level) ? size * size * size : 0.0;
  }

  // Columns per axis of the (x, y) quadrature in inside_volume.
  static constexpr int kColumns = 8;

  // Volume of the cell that lies inside the sphere: exact along z, kColumns x kColumns
  // midpoint quadrature over (x, y). Never exceeds the cell volume.
  MINK_SPHERE_HD double inside_volume(int64_t x, int64_t y, int64_t z, int level) const {
    const double size = cell_size(level);
    const double step = size / kColumns;
    const double z0 = static_cast<double>(z) * size - r;
    const double z1 = z0 + size;
    double length = 0.0;
    for (int i = 0; i < kColumns; ++i) {
      const double dx = static_cast<double>(x) * size + (i + 0.5) * step - r;
      for (int j = 0; j < kColumns; ++j) {
        const double dy = static_cast<double>(y) * size + (j + 0.5) * step - r;
        const double h2 = r * r - dx * dx - dy * dy;
        if (h2 <= 0.0)
          continue;
        const double h = sqrt(h2);
        const double lo = z0 > -h ? z0 : -h;
        const double hi = z1 < h ? z1 : h;
        if (hi > lo)
          length += hi - lo;
      }
    }
    return length * step * step;
  }

  // Estimated volume of the cell whose occupancy disagrees with the sphere: the part outside
  // the sphere for an occupied leaf, the part inside for an empty one. Exactly 0 for cells
  // fully inside or outside (their occupancy is exact).
  MINK_SPHERE_HD double misclassified_volume(int64_t x, int64_t y, int64_t z, int level) const {
    if (!is_boundary(x, y, z, level))
      return 0.0;
    const double size = cell_size(level);
    const double inside = inside_volume(x, y, z, level);
    return occupied(x, y, z, level) ? size * size * size - inside : inside;
  }

  // Accuracy from the misclassified (symmetric-difference) volume.
  MINK_SPHERE_HD double geometric_accuracy(double misclassified) const {
    return 100.0 * (1.0 - misclassified / exact_volume());
  }
};

} // namespace sphere_detail
} // namespace adaptive
} // namespace minkowski

#endif
