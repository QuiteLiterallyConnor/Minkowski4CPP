#ifndef MINKOWSKI_GAUSSIAN_GAUSSIAN_RENDER_HPP
#define MINKOWSKI_GAUSSIAN_GAUSSIAN_RENDER_HPP

#include "gaussian/gaussian_camera.hpp"
#include "gaussian/gaussian_model.hpp"
#include "gaussian/gaussian_tensor.hpp"

#include <torch/torch.h>

namespace minkowski {
namespace gaussian {

struct RenderOptions {
  // [3] background color; undefined means black.
  at::Tensor background;
  // Highest SH degree to evaluate (-1 uses the model's degree).
  int sh_degree = -1;
  // Pixels per tile side (CPU paths; the CUDA kernels always use 16).
  int64_t tile_size = 16;
  // Gaussians composited per chunk inside a tile (LibTorch path only).
  int64_t chunk_size = 1024;
  // Use the fused CUDA rasterizer for CUDA inputs when available.
  bool use_cuda_kernels = true;
  // Use the fused multi-threaded C++ rasterizer for CPU inputs (same
  // algorithm as the CUDA kernels: per-pixel early stop at T < 1e-4).
  // false selects the LibTorch reference path.
  bool use_cpu_kernels = true;
};

struct RenderOutput {
  at::Tensor image;   // [H, W, 3]
  at::Tensor depth;   // [H, W], alpha-weighted camera-space z (0 where empty)
  at::Tensor alpha;   // [H, W], accumulated opacity
  at::Tensor means2d; // [N, 2] screen-space centers; call backward to fill .grad()
  at::Tensor visible; // [N] bool, true if the Gaussian contributed to the image
  at::Tensor radii;   // [N] int64 screen radius in pixels (0 if culled)
  // [] int64 number of (tile, Gaussian) pairs the rasterizer composites; read
  // it with .item() only when needed (it synchronizes CUDA).
  at::Tensor tile_pairs;
};

// Differentiable tile-based alpha-compositing rasterizer written with LibTorch
// operators (CPU and CUDA). Gaussians are projected with the EWA
// approximation, sorted by depth and composited front to back. Gradients flow
// to every input. All tensors must be float32 on the same device.
RenderOutput render_gaussians(const at::Tensor &positions,
                              const at::Tensor &rotations,
                              const at::Tensor &scales,
                              const at::Tensor &opacities,
                              const at::Tensor &colors, const Camera &camera,
                              const RenderOptions &options = {});

// Renders a trainable model, evaluating spherical harmonics for view-dependent
// color.
RenderOutput render(const GaussianModel &model, const Camera &camera,
                    const RenderOptions &options = {});

// Renders an activated tensor using its per-Gaussian colors.
RenderOutput render(const GaussianTensor &gaussians, const Camera &camera,
                    const RenderOptions &options = {});

} // namespace gaussian
} // namespace minkowski

#endif
