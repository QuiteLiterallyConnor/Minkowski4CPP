#include "gaussian/gaussian_trainer.hpp"

#include <algorithm>
#include <cmath>

namespace minkowski {
namespace gaussian {
namespace {

at::Tensor rotation_matrices(const at::Tensor &q) {
  const auto n = q / q.norm(2, {1}, true).clamp_min(1e-12);
  const auto w = n.select(1, 0), x = n.select(1, 1), y = n.select(1, 2),
             z = n.select(1, 3);
  return at::stack(
      {at::stack({1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
                  2 * (x * z + w * y)},
                 1),
       at::stack({2 * (x * y + w * z), 1 - 2 * (x * x + z * z),
                  2 * (y * z - w * x)},
                 1),
       at::stack({2 * (x * z - w * y), 2 * (y * z + w * x),
                  1 - 2 * (x * x + y * y)},
                 1)},
      1);
}

void check_image(const at::Tensor &t, const char *name) {
  TORCH_CHECK(t.dim() == 3 && t.size(2) == 3, name,
              " must have shape [H, W, 3]");
}

} // namespace

at::Tensor l1_loss(const at::Tensor &prediction, const at::Tensor &target) {
  check_image(prediction, "prediction");
  TORCH_CHECK(prediction.sizes() == target.sizes(),
              "prediction and target must have the same shape");
  return (prediction - target).abs().mean();
}

at::Tensor ssim(const at::Tensor &prediction, const at::Tensor &target,
                int64_t window, double sigma) {
  check_image(prediction, "prediction");
  TORCH_CHECK(prediction.sizes() == target.sizes(),
              "prediction and target must have the same shape");
  TORCH_CHECK(window > 0 && window % 2 == 1, "SSIM window must be odd");
  const auto options = prediction.options();
  auto coordinates =
      at::arange(window, options) - static_cast<double>(window / 2);
  auto kernel = at::exp(-(coordinates * coordinates) / (2.0 * sigma * sigma));
  kernel = kernel / kernel.sum();
  const auto window2d = kernel.unsqueeze(1) * kernel.unsqueeze(0);
  const auto weight = window2d.expand({3, 1, window, window}).contiguous();

  const auto x = prediction.permute({2, 0, 1}).unsqueeze(0);
  const auto y = target.permute({2, 0, 1}).unsqueeze(0);
  auto filter = [&](const at::Tensor &t) {
    return at::conv2d(t, weight, {}, 1, window / 2, 1, 3);
  };
  const auto mu_x = filter(x), mu_y = filter(y);
  const auto mu_xx = mu_x * mu_x, mu_yy = mu_y * mu_y, mu_xy = mu_x * mu_y;
  const auto sigma_xx = filter(x * x) - mu_xx;
  const auto sigma_yy = filter(y * y) - mu_yy;
  const auto sigma_xy = filter(x * y) - mu_xy;
  constexpr double c1 = 0.01 * 0.01, c2 = 0.03 * 0.03;
  const auto map = ((2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)) /
                   ((mu_xx + mu_yy + c1) * (sigma_xx + sigma_yy + c2));
  return map.mean();
}

GaussianTrainer::GaussianTrainer(GaussianModel &model,
                                 GaussianTrainerConfig config)
    : model_(&model), config_(std::move(config)) {
  TORCH_CHECK(config_.scene_extent > 0.0, "scene_extent must be positive");
  for (auto &p : model_->enable_gradients()) {
    moment1_.push_back(at::zeros_like(p));
    moment2_.push_back(at::zeros_like(p));
  }
  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(model_->device());
  gradient_accumulator_ = at::zeros({model_->size()}, options);
  gradient_count_ = at::zeros({model_->size()}, options);
}

int GaussianTrainer::active_sh_degree() const {
  if (config_.sh_increase_interval <= 0)
    return model_->sh_degree();
  const auto grown = static_cast<int64_t>(
      std::min<int64_t>(3, iteration_ / config_.sh_increase_interval));
  return static_cast<int>(std::min<int64_t>(grown, model_->sh_degree()));
}

double GaussianTrainer::learning_rate(size_t parameter) const {
  switch (parameter) {
  case 0: {
    const double t = config_.lr_decay_steps > 0
                         ? std::min(1.0, static_cast<double>(iteration_) /
                                             config_.lr_decay_steps)
                         : 1.0;
    const double lr = std::exp((1.0 - t) * std::log(config_.lr_position) +
                               t * std::log(config_.lr_position_final));
    return lr * config_.scene_extent;
  }
  case 1: return config_.lr_dc;
  case 2: return config_.lr_rest;
  case 3: return config_.lr_opacity;
  case 4: return config_.lr_scale;
  default: return config_.lr_rotation;
  }
}

void GaussianTrainer::adam_step() {
  torch::NoGradGuard no_grad;
  ++adam_steps_;
  const double bias1 = 1.0 - std::pow(config_.beta1, adam_steps_);
  const double bias2 = 1.0 - std::pow(config_.beta2, adam_steps_);
  const auto parameters = model_->parameters();
  for (size_t i = 0; i < parameters.size(); ++i) {
    const auto &p = parameters[i];
    const auto &g = p.grad();
    if (!g.defined() || p.numel() == 0)
      continue;
    moment1_[i].mul_(config_.beta1).add_(g, 1.0 - config_.beta1);
    moment2_[i].mul_(config_.beta2).addcmul_(g, g, 1.0 - config_.beta2);
    const auto denominator =
        (moment2_[i] / bias2).sqrt().add_(config_.epsilon);
    p.addcdiv_(moment1_[i], denominator, -learning_rate(i) / bias1);
  }
}

at::Tensor GaussianTrainer::render_image(const Camera &camera) const {
  torch::NoGradGuard no_grad;
  RenderOptions options = config_.render;
  options.sh_degree = active_sh_degree();
  return render(*model_, camera, options).image;
}

void GaussianTrainer::accumulate_statistics(const RenderOutput &output,
                                            const Camera &camera) {
  torch::NoGradGuard no_grad;
  const auto &grad = output.means2d.grad();
  if (!grad.defined())
    return;
  const auto scaled = at::stack(
      {grad.select(1, 0) * (0.5 * static_cast<double>(camera.width)),
       grad.select(1, 1) * (0.5 * static_cast<double>(camera.height))},
      1);
  const auto norm = scaled.norm(2, {1});
  const auto visible = output.visible.to(at::kFloat);
  gradient_accumulator_ = gradient_accumulator_ + norm * visible;
  gradient_count_ = gradient_count_ + visible;
}

GaussianTrainer::StepResult GaussianTrainer::step(const Camera &camera,
                                                  const at::Tensor &target) {
  check_image(target, "target");
  TORCH_CHECK(target.size(0) == camera.height && target.size(1) == camera.width,
              "target size must match the camera");
  const auto device = model_->device();
  const auto image_target = target.to(device, at::kFloat);

  RenderOptions options = config_.render;
  options.sh_degree = active_sh_degree();
  auto output = render(*model_, camera, options);

  at::Tensor loss = (1.0 - config_.lambda_ssim) *
                    gaussian::l1_loss(output.image, image_target);
  if (config_.lambda_ssim > 0.0)
    loss = loss + config_.lambda_ssim *
                      (1.0 - gaussian::ssim(output.image, image_target));

  for (auto &p : model_->parameters())
    p.mutable_grad() = at::Tensor();
  loss.backward();
  accumulate_statistics(output, camera);
  adam_step();
  ++iteration_;

  if (config_.densify && iteration_ < config_.densify_stop &&
      iteration_ > config_.densify_start && config_.densify_interval > 0 &&
      iteration_ % config_.densify_interval == 0)
    densify_and_prune();
  if (config_.densify && config_.opacity_reset_interval > 0 &&
      iteration_ < config_.densify_stop &&
      iteration_ % config_.opacity_reset_interval == 0)
    reset_opacity();

  StepResult result;
  result.loss = loss.item<double>();
  result.gaussians = model_->size();
  result.iteration = iteration_;
  return result;
}

void GaussianTrainer::apply_rows(const std::vector<at::Tensor> &new_rows,
                                 const at::Tensor &keep) {
  torch::NoGradGuard no_grad;
  const auto keep_indices = keep.nonzero().squeeze(1);
  const auto parameters = model_->parameters();
  std::vector<at::Tensor> updated;
  for (size_t i = 0; i < parameters.size(); ++i) {
    updated.push_back(at::cat({parameters[i].detach(), new_rows[i]}, 0)
                          .index_select(0, keep_indices)
                          .contiguous());
    moment1_[i] = at::cat({moment1_[i], at::zeros_like(new_rows[i])}, 0)
                      .index_select(0, keep_indices)
                      .contiguous();
    moment2_[i] = at::cat({moment2_[i], at::zeros_like(new_rows[i])}, 0)
                      .index_select(0, keep_indices)
                      .contiguous();
  }
  model_->set_parameters(updated);
  model_->enable_gradients();
  const auto options =
      at::TensorOptions().dtype(at::kFloat).device(model_->device());
  gradient_accumulator_ = at::zeros({model_->size()}, options);
  gradient_count_ = at::zeros({model_->size()}, options);
}

std::pair<int64_t, int64_t> GaussianTrainer::densify_and_prune() {
  torch::NoGradGuard no_grad;
  const int64_t n = model_->size();
  const auto parameters = model_->parameters();
  const auto gradients =
      gradient_accumulator_ / gradient_count_.clamp_min(1.0);
  const auto scales = model_->scales().detach();
  const auto max_scale = std::get<0>(scales.max(1));
  const auto selected = gradients >= config_.densify_grad_threshold;
  const double size_threshold = config_.percent_dense * config_.scene_extent;
  auto clone_mask = selected & (max_scale <= size_threshold);
  auto split_mask = selected & (max_scale > size_threshold);

  int64_t clones = clone_mask.sum().item<int64_t>();
  int64_t splits = split_mask.sum().item<int64_t>();
  if (config_.max_gaussians > 0 &&
      n + clones + 2 * splits > config_.max_gaussians) {
    clone_mask = at::zeros_like(clone_mask);
    split_mask = at::zeros_like(split_mask);
    clones = splits = 0;
  }

  const auto clone_indices = clone_mask.nonzero().squeeze(1);
  const auto split_indices = split_mask.nonzero().squeeze(1);
  at::Tensor split_xyz, split_scale_raw;
  if (splits > 0) {
    // Two children sampled from the parent's Gaussian, each 1.6x smaller.
    const auto repeated_scales =
        scales.index_select(0, split_indices).repeat({2, 1});
    const auto rotation = rotation_matrices(
        model_->rotation_raw().detach().index_select(0, split_indices)
            .repeat({2, 1}));
    const auto noise = at::randn_like(repeated_scales) * repeated_scales;
    split_xyz = at::bmm(rotation, noise.unsqueeze(-1)).squeeze(-1) +
                model_->xyz().detach().index_select(0, split_indices)
                    .repeat({2, 1});
    split_scale_raw = at::log(repeated_scales / 1.6);
  }

  std::vector<at::Tensor> new_rows;
  for (size_t i = 0; i < parameters.size(); ++i) {
    std::vector<at::Tensor> parts{
        parameters[i].detach().index_select(0, clone_indices)};
    if (splits > 0) {
      if (i == 0) {
        parts.push_back(split_xyz);
      } else if (i == 4) {
        parts.push_back(split_scale_raw);
      } else {
        const auto rows = parameters[i].detach().index_select(0, split_indices);
        parts.push_back(at::cat({rows, rows}, 0));
      }
    }
    new_rows.push_back(at::cat(parts, 0));
  }

  const int64_t added = clones + 2 * splits;
  const auto opacities = model_->opacities().detach().reshape({-1});
  auto prune = (opacities < config_.min_opacity) |
               (max_scale > config_.max_scale_fraction * config_.scene_extent) |
               split_mask;
  auto prune_all = at::cat({prune, at::zeros({added}, prune.options())}, 0);
  auto keep = ~prune_all;
  if (keep.sum().item<int64_t>() == 0) {
    keep = at::ones_like(keep);
    prune_all = at::zeros_like(prune_all);
  }
  const int64_t removed = prune_all.sum().item<int64_t>();
  apply_rows(new_rows, keep);
  return {added, removed};
}

void GaussianTrainer::reset_opacity() {
  torch::NoGradGuard no_grad;
  const auto &raw = model_->opacity_raw();
  const auto clipped = at::sigmoid(raw).clamp_max(0.01);
  raw.copy_(at::log(clipped / (1.0 - clipped)));
  moment1_[3].zero_();
  moment2_[3].zero_();
}

} // namespace gaussian
} // namespace minkowski
