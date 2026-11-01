---
title: Rendering & training
description: Differentiable Gaussian rasterization, cameras, spherical harmonics, and 3DGS-style optimization.
permalink: /rendering/
---

## Trainable model

`GaussianModel` stores raw position, quaternion, log-scale, logit-opacity,
DC color, and SH-rest parameters. Activated views expose valid scales,
opacities, and rotations. SH degrees **0 through 3** are supported.

Construct from a point cloud with `GaussianModel::from_points(points, colors)`
or load a scene with `load_gaussian_model_ply`. Models also support `select`,
`clone`, and `to`.

```cpp
using namespace minkowski::gaussian;
auto model = load_gaussian_model_ply("scene.ply");
auto gpu_model = model.to(torch::Device(torch::kCUDA, 0));
```

Keep the trained model object alive for the entire trainer lifetime.

## Camera convention

The pinhole camera uses an **OpenCV world-to-camera** convention: +x right,
+y down, +z forward, with pixel centers at `(i+0.5, j+0.5)`. Its matrix
is a float `[4,4]` tensor. Camera helpers keep the matrix on CPU; the renderer
converts it to its input tensor options.
`Camera::from_fov` and `Camera::look_at` are convenience constructors.

```cpp
Camera camera = Camera::look_at(
    {0, 0, -4}, {0, 0, 0}, {0, -1, 0},
    /*width=*/512, /*height=*/512, /*fov_x=*/1.0);
```

The field of view is in radians. Supply calibrated cameras for actual
training rather than substituting this illustrative camera.

## Render a scene

```cpp
auto model = load_gaussian_model_ply("scene.ply");
auto output = render(model, camera);
auto image = output.image;
```

Overloads render either a `GaussianModel` (with SH-based view-dependent
color) or a `GaussianTensor` (with per-Gaussian RGB).
`render_gaussians` accepts the activated position, rotation, scale, opacity,
and color tensors directly.

All render input tensors must be **float32 on the same device**.

| Output | Shape | Meaning |
|---|---|---|
| `image` | `[H,W,3]` | Rendered RGB |
| `depth` | `[H,W]` | Alpha-weighted camera-space z; zero where empty |
| `alpha` | `[H,W]` | Accumulated opacity |
| `means2d` | `[N,2]` | Screen-space centers, used for gradient statistics |
| `visible` | `[N]`, bool | Whether a Gaussian contributed |
| `radii` | `[N]`, int64 | Screen radius in pixels; zero if culled |

Depth is alpha-weighted, not automatically a normalized expected-depth
map. Interpret it accordingly when comparing against sensor depth.

## Rasterizer paths

The renderer uses EWA projection, depth sorting, and front-to-back alpha
compositing in tiles. Autograd reaches Gaussian parameters.

| Path | When used | Behavior |
|---|---|---|
| Fused CUDA | CUDA tensors, kernels available, `use_cuda_kernels=true` | Forward/backward compositing kernels, 16x16 tiles |
| LibTorch | CPU, or CUDA with fused kernels disabled | Differentiable tensor-based fallback |

Projection and SH evaluation remain LibTorch operations. CUDA tile
compositing uses shared-memory batches and warp-reduced gradient
accumulation. The fused path **always uses tile size 16** and ignores
`tile_size` / `chunk_size` tuning.

```cpp
RenderOptions options;
options.use_cuda_kernels = false;
options.tile_size = 16;
options.chunk_size = 1024;
options.sh_degree = -1;
auto reference = render(model, camera, options);
```

`sh_degree=-1` uses the model's degree. An undefined background means black;
otherwise supply a `[3]` background tensor on the appropriate device.

## Train against posed images

The application supplies views: each is a camera and a float32 target image
of shape `[H,W,3]` with values in `[0,1]`.

```cpp
#include <minkowski.hpp>
#include <utility>
#include <vector>
using namespace minkowski::gaussian;

void train_scene(GaussianModel& model,
                 const std::vector<std::pair<Camera, at::Tensor>>& views) {
  GaussianTrainerConfig config;
  config.scene_extent = 4.0;
  config.max_gaussians = 1000000;
  GaussianTrainer trainer(model, config);
  for (const auto& [camera, target] : views) {
    auto result = trainer.step(camera, target);
    // result.loss, result.gaussians, and result.iteration are scalar metadata.
  }
  save_gaussian_ply(model, "trained.ply");
}
```

This function is one pass through already-loaded views, not a complete
dataset loader. Repeat or sample views as appropriate for your training
schedule. Keep target tensors compatible with the model and camera.

`step` performs loss evaluation, backward, Adam updates, statistics
collection, scheduled densification/pruning, and opacity reset.
`render_image(camera)` renders without tracking gradients.

## Trainer configuration

| Setting | Default | Purpose |
|---|---|---|
| `lr_position` / `lr_position_final` | `1.6e-4` / `1.6e-6` | Position LR endpoints, scaled by scene extent |
| `lr_decay_steps` | `30000` | Exponential position-LR schedule |
| `lr_dc` / `lr_rest` | `2.5e-3` / `1.25e-4` | Color/SH learning rates |
| `lr_opacity` / `lr_scale` / `lr_rotation` | `0.05` / `5e-3` / `1e-3` | Remaining parameter learning rates |
| `lambda_ssim` | `0.2` | SSIM contribution to L1 + SSIM reconstruction |
| `sh_increase_interval` | `1000` | Grow active SH degree |
| `densify` | `true` | Enable clone/split and prune schedule |
| `densify_start` / `densify_stop` | `500` / `15000` | Densification window |
| `densify_interval` | `100` | Densification cadence |
| `densify_grad_threshold` | `2e-4` | Screen-space gradient threshold |
| `percent_dense` | `0.01` | Size cutoff between cloning and splitting |
| `min_opacity` | `0.005` | Prune transparent Gaussians |
| `max_scale_fraction` | `0.1` | Prune overly large Gaussians |
| `opacity_reset_interval` | `3000` | Reset opacity cadence |
| `max_gaussians` | `0` | Count bound; zero means unlimited |
| `scene_extent` | `1.0` | Scene size for position LR and size thresholds |

High-gradient small Gaussians are cloned; large ones are split.
The trainer's Adam state stays aligned when parameter rows are replaced.
For long runs, set a count bound and monitor memory rather than assuming
the default unlimited growth is safe.

`densify_and_prune()` returns `{added, removed}`.
`reset_opacity()` clamps opacities to at most 0.01 and clears their Adam
state. `accumulate_statistics` is exposed for custom loops; standard
`step` already calls it.

See [tests](testing.md) for renderer gradient checks and
[benchmarks](testing.md#reported-local-measurements) for fused/fallback timings.
