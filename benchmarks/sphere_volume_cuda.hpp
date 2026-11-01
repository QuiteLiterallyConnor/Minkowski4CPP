#ifndef SPHERE_VOLUME_CUDA_HPP
#define SPHERE_VOLUME_CUDA_HPP

#include <cstdint>
#include <vector>

namespace sphere_cuda {

// False when no CUDA device is present; callers then use the host paths.
bool available();

// Same refinement rule as the host generator: boundary cells are split one level at a time
// until another split would exceed `budget` voxels. coords is [N, 3] x/y/z, levels is [N].
void generate_adaptive(int64_t budget, uint32_t max_depth, double side, double radius,
                       std::vector<int64_t>& coords, std::vector<int32_t>& levels);

// Cells in x-major order of an n x n x n grid, all at level 0.
void generate_uniform(int n, std::vector<int64_t>& coords, std::vector<int32_t>& levels);

struct Estimate {
  double volume = 0.0;
  int64_t inside = 0;   // center inside the sphere
  int64_t boundary = 0; // box straddles the surface
};

// Sphere-volume estimate over the given leaves (cell centers decide inside/outside).
Estimate estimate(const int64_t* coords, const int32_t* levels, int64_t n, double base_size,
                  double radius);

} // namespace sphere_cuda

#endif
