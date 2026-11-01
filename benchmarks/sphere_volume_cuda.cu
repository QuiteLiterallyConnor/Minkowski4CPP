#include "sphere_volume_cuda.hpp"

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace sphere_cuda {
namespace {

struct Cell {
  int64_t x, y, z;
  int32_t level;
  int32_t pad;
};

enum Relation { kOutside = 0, kInside = 1, kBoundary = 2 };

__host__ __device__ inline double cell_size(double base, int32_t level) {
  return base * scalbn(1.0, -level);
}

// Mirrors classify_cell in sphere_volume.cpp.
__host__ __device__ inline int classify(const Cell &c, double base, double radius) {
  const double size = cell_size(base, c.level);
  const int64_t idx[3] = {c.x, c.y, c.z};
  double nearest_sq = 0.0, farthest_sq = 0.0;
  for (int a = 0; a < 3; ++a) {
    const double lo = static_cast<double>(idx[a]) * size;
    const double hi = lo + size;
    const double nearest = fmax(lo, fmin(radius, hi));
    const double nd = nearest - radius;
    const double fd = fmax(fabs(lo - radius), fabs(hi - radius));
    nearest_sq += nd * nd;
    farthest_sq += fd * fd;
  }
  if (nearest_sq >= radius * radius)
    return kOutside;
  if (farthest_sq <= radius * radius)
    return kInside;
  return kBoundary;
}

struct Truthy {
  __host__ __device__ bool operator()(int v) const { return v != 0; }
};
struct Falsy {
  __host__ __device__ bool operator()(int v) const { return v == 0; }
};

__global__ void split_kernel(const int *flag, const int *rank, int64_t n, int64_t k, int *split) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    split[i] = flag[i] && rank[i] < k;
}

// Child order (x slowest, z fastest) matches child_cell in the host generator.
__global__ void children_kernel(const Cell *parents, int64_t n, double side, double radius,
                                Cell *out, int *flag) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n * 8)
    return;
  const Cell p = parents[i / 8];
  const int child = static_cast<int>(i % 8);
  Cell c;
  c.x = 2 * p.x + ((child >> 2) & 1);
  c.y = 2 * p.y + ((child >> 1) & 1);
  c.z = 2 * p.z + (child & 1);
  c.level = p.level + 1;
  c.pad = 0;
  out[i] = c;
  flag[i] = classify(c, side, radius) == kBoundary;
}

__global__ void export_kernel(const Cell *cells, int64_t n, int64_t *coords, int32_t *levels) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  coords[i * 3] = cells[i].x;
  coords[i * 3 + 1] = cells[i].y;
  coords[i * 3 + 2] = cells[i].z;
  levels[i] = cells[i].level;
}

__global__ void uniform_kernel(int n, int64_t *coords, int32_t *levels) {
  const int64_t total = static_cast<int64_t>(n) * n * n;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total)
    return;
  coords[i * 3] = i / (static_cast<int64_t>(n) * n);
  coords[i * 3 + 1] = (i / n) % n;
  coords[i * 3 + 2] = i % n;
  levels[i] = 0;
}

struct Acc {
  double volume;
  int64_t inside;
  int64_t boundary;
};

struct AccAdd {
  __host__ __device__ Acc operator()(const Acc &a, const Acc &b) const {
    return {a.volume + b.volume, a.inside + b.inside, a.boundary + b.boundary};
  }
};

struct EvalCell {
  const int64_t *coords;
  const int32_t *levels;
  double base, radius;
  __host__ __device__ Acc operator()(int64_t i) const {
    const Cell c{coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2], levels[i], 0};
    const double size = cell_size(base, c.level);
    const double dx = (static_cast<double>(c.x) + 0.5) * size - radius;
    const double dy = (static_cast<double>(c.y) + 0.5) * size - radius;
    const double dz = (static_cast<double>(c.z) + 0.5) * size - radius;
    Acc a{0.0, 0, 0};
    if (dx * dx + dy * dy + dz * dz <= radius * radius) {
      a.volume = size * size * size;
      a.inside = 1;
    }
    a.boundary = classify(c, base, radius) == kBoundary;
    return a;
  }
};

constexpr int kThreads = 256;
int blocks_for(int64_t n) { return static_cast<int>((n + kThreads - 1) / kThreads); }

void check(cudaError_t e, const char *what) {
  if (e != cudaSuccess)
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

void download(const thrust::device_vector<Cell> &cells, std::vector<int64_t> &coords,
              std::vector<int32_t> &levels) {
  const int64_t n = static_cast<int64_t>(cells.size());
  thrust::device_vector<int64_t> d_coords(static_cast<size_t>(n) * 3);
  thrust::device_vector<int32_t> d_levels(static_cast<size_t>(n));
  export_kernel<<<blocks_for(n), kThreads>>>(thrust::raw_pointer_cast(cells.data()), n,
                                             thrust::raw_pointer_cast(d_coords.data()),
                                             thrust::raw_pointer_cast(d_levels.data()));
  check(cudaGetLastError(), "export kernel");
  coords.resize(static_cast<size_t>(n) * 3);
  levels.resize(static_cast<size_t>(n));
  check(cudaMemcpy(coords.data(), thrust::raw_pointer_cast(d_coords.data()),
                   coords.size() * sizeof(int64_t), cudaMemcpyDeviceToHost),
        "coords download");
  check(cudaMemcpy(levels.data(), thrust::raw_pointer_cast(d_levels.data()),
                   levels.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
        "levels download");
}

} // namespace

bool available() {
  static const bool ok = [] {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
  }();
  return ok;
}

void generate_adaptive(int64_t budget, uint32_t max_depth, double side, double radius,
                       std::vector<int64_t> &coords, std::vector<int32_t> &levels) {
  const Cell root{0, 0, 0, 0, 0};
  thrust::device_vector<Cell> cur(1, root);
  thrust::device_vector<int> flag(1, classify(root, side, radius) == kBoundary ? 1 : 0);
  thrust::device_vector<Cell> leaves;
  int64_t count = 1;
  while (!cur.empty()) {
    const int64_t n = static_cast<int64_t>(cur.size());
    thrust::device_vector<int> rank(static_cast<size_t>(n));
    thrust::exclusive_scan(thrust::device, flag.begin(), flag.end(), rank.begin());
    const int64_t boundary = static_cast<int64_t>(rank[n - 1]) + flag[n - 1];
    const Cell first = cur[0];
    int64_t k = 0;
    if (first.level < static_cast<int32_t>(max_depth) && budget > count)
      k = std::min<int64_t>(boundary, (budget - count) / 7);

    thrust::device_vector<int> split(static_cast<size_t>(n));
    split_kernel<<<blocks_for(n), kThreads>>>(thrust::raw_pointer_cast(flag.data()),
                                              thrust::raw_pointer_cast(rank.data()), n, k,
                                              thrust::raw_pointer_cast(split.data()));
    check(cudaGetLastError(), "split kernel");
    const size_t old = leaves.size();
    leaves.resize(old + static_cast<size_t>(n - k));
    thrust::copy_if(thrust::device, cur.begin(), cur.end(), split.begin(), leaves.begin() + old,
                    Falsy{});
    if (k == 0)
      break;

    thrust::device_vector<Cell> parents(static_cast<size_t>(k));
    thrust::copy_if(thrust::device, cur.begin(), cur.end(), split.begin(), parents.begin(),
                    Truthy{});
    count += 7 * k;
    thrust::device_vector<Cell> next(static_cast<size_t>(8 * k));
    thrust::device_vector<int> next_flag(static_cast<size_t>(8 * k));
    children_kernel<<<blocks_for(8 * k), kThreads>>>(
        thrust::raw_pointer_cast(parents.data()), k, side, radius,
        thrust::raw_pointer_cast(next.data()), thrust::raw_pointer_cast(next_flag.data()));
    check(cudaGetLastError(), "children kernel");
    if (k < boundary) {
      const size_t at = leaves.size();
      leaves.resize(at + next.size());
      thrust::copy(thrust::device, next.begin(), next.end(), leaves.begin() + at);
      break;
    }
    cur.swap(next);
    flag.swap(next_flag);
  }
  download(leaves, coords, levels);
}

void generate_uniform(int n, std::vector<int64_t> &coords, std::vector<int32_t> &levels) {
  const int64_t total = static_cast<int64_t>(n) * n * n;
  thrust::device_vector<int64_t> d_coords(static_cast<size_t>(total) * 3);
  thrust::device_vector<int32_t> d_levels(static_cast<size_t>(total));
  uniform_kernel<<<blocks_for(total), kThreads>>>(n, thrust::raw_pointer_cast(d_coords.data()),
                                                  thrust::raw_pointer_cast(d_levels.data()));
  check(cudaGetLastError(), "uniform kernel");
  coords.resize(static_cast<size_t>(total) * 3);
  levels.resize(static_cast<size_t>(total));
  check(cudaMemcpy(coords.data(), thrust::raw_pointer_cast(d_coords.data()),
                   coords.size() * sizeof(int64_t), cudaMemcpyDeviceToHost),
        "coords download");
  check(cudaMemcpy(levels.data(), thrust::raw_pointer_cast(d_levels.data()),
                   levels.size() * sizeof(int32_t), cudaMemcpyDeviceToHost),
        "levels download");
}

Estimate estimate(const int64_t *coords, const int32_t *levels, int64_t n, double base_size,
                  double radius) {
  Estimate out;
  if (n == 0)
    return out;
  thrust::device_vector<int64_t> d_coords(coords, coords + n * 3);
  thrust::device_vector<int32_t> d_levels(levels, levels + n);
  const auto idx = thrust::make_counting_iterator<int64_t>(0);
  const Acc acc = thrust::transform_reduce(
      thrust::device, idx, idx + n,
      EvalCell{thrust::raw_pointer_cast(d_coords.data()),
               thrust::raw_pointer_cast(d_levels.data()), base_size, radius},
      Acc{0.0, 0, 0}, AccAdd{});
  out.volume = acc.volume;
  out.inside = acc.inside;
  out.boundary = acc.boundary;
  return out;
}

} // namespace sphere_cuda
