#ifndef MINKOWSKI_GAUSSIAN_GAUSSIAN_TRAINER_HPP
#define MINKOWSKI_GAUSSIAN_GAUSSIAN_TRAINER_HPP

#include "gaussian/gaussian_camera.hpp"
#include "gaussian/gaussian_model.hpp"
#include "gaussian/gaussian_render.hpp"

#include <torch/torch.h>

#include <cstdint>
#include <vector>

namespace minkowski {
namespace gaussian {

// Mean absolute error and structural similarity for [H, W, 3] images.
at::Tensor l1_loss(const at::Tensor &prediction, const at::Tensor &target);
at::Tensor ssim(const at::Tensor &prediction, const at::Tensor &target,
                int64_t window = 11, double sigma = 1.5);

struct GaussianTrainerConfig {
  // Learning rates (the position rate is multiplied by scene_extent and decays
  // exponentially from lr_position to lr_position_final over lr_decay_steps).
  double lr_position = 1.6e-4;
  double lr_position_final = 1.6e-6;
  int64_t lr_decay_steps = 30000;
  double lr_dc = 2.5e-3;
  double lr_rest = 1.25e-4;
  double lr_opacity = 0.05;
  double lr_scale = 5e-3;
  double lr_rotation = 1e-3;
  double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-15;

  double lambda_ssim = 0.2;
  // The active SH degree grows by one every sh_increase_interval iterations.
  int64_t sh_increase_interval = 1000;

  // Densification (adaptive split / clone) and pruning.
  bool densify = true;
  int64_t densify_start = 500;
  int64_t densify_interval = 100;
  int64_t densify_stop = 15000;
  double densify_grad_threshold = 2e-4;
  // Gaussians larger than percent_dense * scene_extent are split, others cloned.
  double percent_dense = 0.01;
  double min_opacity = 0.005;
  // Gaussians whose largest scale exceeds this fraction of the extent are pruned.
  double max_scale_fraction = 0.1;
  int64_t opacity_reset_interval = 3000;
  // Upper bound on the number of Gaussians (0 = unlimited).
  int64_t max_gaussians = 0;
  // Characteristic size of the scene (e.g. camera-rig radius).
  double scene_extent = 1.0;

  RenderOptions render;
};

// Trains a GaussianModel against posed images with the 3DGS recipe: L1 + SSIM
// loss, Adam with per-parameter learning rates, SH-degree warm-up, periodic
// densification and pruning, and opacity resets. The model must outlive the
// trainer; the trainer replaces the model's parameter tensors when the number
// of Gaussians changes.
class GaussianTrainer {
public:
  struct StepResult {
    double loss = 0.0;
    int64_t gaussians = 0;
    int64_t iteration = 0;
  };

  GaussianTrainer(GaussianModel &model, GaussianTrainerConfig config = {});

  // One optimization step on a single posed image ([H, W, 3] float32 target).
  StepResult step(const Camera &camera, const at::Tensor &target);

  // Renders without tracking gradients (using the active SH degree).
  at::Tensor render_image(const Camera &camera) const;

  int64_t iteration() const { return iteration_; }
  int active_sh_degree() const;
  const GaussianTrainerConfig &config() const { return config_; }

  // Accumulates the screen-space positional gradient of the visible Gaussians
  // (as in 3DGS, in normalized device units). Called by step(); exposed for
  // custom loops and testing.
  void accumulate_statistics(const RenderOutput &output, const Camera &camera);

  // Clones small and splits large Gaussians with a high accumulated gradient,
  // then prunes transparent or oversized ones. Keeps Adam state consistent.
  // Returns {added, removed}.
  std::pair<int64_t, int64_t> densify_and_prune();

  // Clamps all opacities to at most 0.01 and clears their Adam state.
  void reset_opacity();

private:
  double learning_rate(size_t parameter) const;
  void adam_step();
  void apply_rows(const std::vector<at::Tensor> &new_rows,
                  const at::Tensor &keep);

  GaussianModel *model_;
  GaussianTrainerConfig config_;
  int64_t iteration_ = 0;
  int64_t adam_steps_ = 0;
  std::vector<at::Tensor> moment1_, moment2_;
  at::Tensor gradient_accumulator_; // [N]
  at::Tensor gradient_count_;       // [N]
};

} // namespace gaussian
} // namespace minkowski

#endif
