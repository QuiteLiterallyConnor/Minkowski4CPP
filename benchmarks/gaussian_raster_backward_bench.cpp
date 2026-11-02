// A direct raster backward benchmark isolates traversal work from projection,
// sorting, optimizer state and allocator resets. Synthetic controls intentionally
// vary how much of a tile's depth list each two-row warp actually reaches.
#include "gaussian/gaussian_cuda.hpp"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace gd = minkowski::gaussian::detail;

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return (v[(v.size() - 1) / 2] + v[v.size() / 2]) * 0.5;
}

int main(int argc, char **argv) {
  try {
    int tile_count = 32, per_tile = 256, warmup = 5, iterations = 30;
    std::string pattern = "row-groups", output = "raster-backward.json";
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      auto next = [&]() {
        if (++i == argc) throw std::invalid_argument("missing option value");
        return std::string(argv[i]);
      };
      if (arg == "--tiles") tile_count = std::stoi(next());
      else if (arg == "--per-tile") per_tile = std::stoi(next());
      else if (arg == "--warmup") warmup = std::stoi(next());
      else if (arg == "--iterations") iterations = std::stoi(next());
      else if (arg == "--pattern") pattern = next();
      else if (arg == "--parallel-atomics") gd::cuda_raster_backward_parallel_atomics() = std::stoi(next()) != 0;
      else if (arg == "--output") output = next();
      else throw std::invalid_argument("unknown option: " + arg);
    }
    TORCH_CHECK(tile_count > 0 && tile_count <= 128 && per_tile >= 8 &&
                    per_tile <= 4096 && iterations > 0 && warmup >= 0,
                "invalid benchmark dimensions");
    TORCH_CHECK(pattern == "row-groups" || pattern == "dense" || pattern == "sparse",
                "unknown pattern");
    const int width = tile_count * 16, height = width, tiles = tile_count * tile_count;
    const int n = tiles * per_tile;
    auto attrs_cpu = at::empty({n, 10}, at::kFloat);
    auto ids_cpu = at::arange(n, at::kInt);
    auto ranges_cpu = at::empty({tiles, 2}, at::kInt);
    auto *a = attrs_cpu.data_ptr<float>();
    auto *ranges = ranges_cpu.data_ptr<int32_t>();
    for (int tile = 0; tile < tiles; ++tile) {
      ranges[2 * tile] = tile * per_tile;
      ranges[2 * tile + 1] = (tile + 1) * per_tile;
      for (int j = 0; j < per_tile; ++j) {
        float *v = a + static_cast<int64_t>(tile * per_tile + j) * 10;
        v[0] = (tile % tile_count) * 16 + (pattern == "sparse" ? 4.f : 8.f);
        v[1] = (tile / tile_count) * 16 + (pattern == "row-groups" ? 2.f * (8 * j / per_tile) + 1.f :
                                          pattern == "sparse" ? 4.f : 8.f);
        v[2] = pattern == "sparse" ? 4.f : 0.015f;
        v[3] = 0.f;
        v[4] = pattern == "row-groups" ? 4.f : v[2];
        v[5] = pattern == "row-groups" ? 0.8f : 0.05f;
        v[6] = 1.f + 0.001f * j;
        v[7] = 0.1f + 0.8f * (j % 17) / 17.f;
        v[8] = 0.2f + 0.7f * (j % 11) / 11.f;
        v[9] = 0.3f + 0.6f * (j % 7) / 7.f;
      }
    }
    const auto opts = at::TensorOptions().device(at::kCUDA).dtype(at::kFloat);
    auto attrs = attrs_cpu.to(at::kCUDA), ids = ids_cpu.to(at::kCUDA);
    auto tile_ranges = ranges_cpu.to(at::kCUDA);
    auto bg = at::tensor({0.1f, 0.2f, 0.3f}).to(at::kCUDA);
    auto out = at::empty({height, width, 5}, opts);
    auto trans = at::empty({height, width}, opts);
    auto last = at::empty({height, width}, opts.dtype(at::kInt));
    auto go = (at::arange(height * width * 5, opts).remainder(23) / 23 - 0.4).view({height, width, 5});
    auto grad = at::zeros_like(attrs);
    gd::rasterize_forward_cuda(attrs, ids, tile_ranges, bg, height, width, out, trans, last);
    // Count loop positions of the present tile-wide bound and a prospective
    // per-warp bound. This instrumentation is outside every timed interval.
    const auto last_cpu = last.cpu();
    const auto *lp = last_cpu.data_ptr<int32_t>();
    int64_t block_scans = 0, warp_scans = 0;
    for (int tile = 0; tile < tiles; ++tile) {
      int block_max = 0;
      for (int warp = 0; warp < 8; ++warp) {
        int warp_max = 0;
        for (int lane = 0; lane < 32; ++lane) {
          const int x = (tile % tile_count) * 16 + lane % 16;
          const int y = (tile / tile_count) * 16 + warp * 2 + lane / 16;
          warp_max = std::max(warp_max, lp[y * width + x]);
        }
        block_max = std::max(block_max, warp_max);
        warp_scans += warp_max;
      }
      block_scans += 8 * block_max;
    }
    cudaEvent_t begin, end;
    C10_CUDA_CHECK(cudaEventCreate(&begin));
    C10_CUDA_CHECK(cudaEventCreate(&end));
    const auto stream = at::cuda::getCurrentCUDAStream();
    std::vector<double> gpu_ms, wall_ms;
    for (int i = 0; i < warmup + iterations; ++i) {
      grad.zero_();
      C10_CUDA_CHECK(cudaStreamSynchronize(stream));
      auto wall = std::chrono::steady_clock::now();
      C10_CUDA_CHECK(cudaEventRecord(begin, stream));
      gd::rasterize_backward_cuda(attrs, ids, tile_ranges, bg, go, trans, last, height, width, grad);
      C10_CUDA_CHECK(cudaEventRecord(end, stream));
      C10_CUDA_CHECK(cudaEventSynchronize(end));
      float elapsed = 0;
      C10_CUDA_CHECK(cudaEventElapsedTime(&elapsed, begin, end));
      if (i >= warmup) {
        gpu_ms.push_back(elapsed);
        wall_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall).count());
      }
    }
    C10_CUDA_CHECK(cudaEventDestroy(begin));
    C10_CUDA_CHECK(cudaEventDestroy(end));
    std::ofstream file(output);
    TORCH_CHECK(file.good(), "cannot open benchmark output");
    file << "{\n  \"pattern\": \"" << pattern << "\", \"tiles\": " << tiles
         << ", \"per_tile\": " << per_tile << ", \"warmup\": " << warmup
         << ", \"iterations\": " << iterations << ",\n  \"block_scans\": " << block_scans
         << ", \"warp_scans\": " << warp_scans << ", \"gpu_median_ms\": " << median(gpu_ms)
         << ", \"wall_median_ms\": " << median(wall_ms) << ",\n  \"gpu_ms\": [";
    for (size_t i = 0; i < gpu_ms.size(); ++i) file << (i ? ", " : "") << gpu_ms[i];
    file << "],\n  \"wall_ms\": [";
    for (size_t i = 0; i < wall_ms.size(); ++i) file << (i ? ", " : "") << wall_ms[i];
    file << "]\n}\n";
    std::cout << pattern << ": tile scans " << block_scans << ", warp scans " << warp_scans
              << ", GPU backward " << median(gpu_ms) << " ms\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
  return 0;
}
