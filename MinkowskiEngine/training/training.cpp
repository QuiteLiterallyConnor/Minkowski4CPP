
#include "training.hpp"

namespace minkowski {
namespace training {

std::unique_ptr<torch::optim::Optimizer>
create_optimizer(const std::vector<torch::Tensor> &parameters,
                 const OptimizerConfig &config) {
  switch (config.type) {
  case OptimizerType::SGD: {
    auto opts = torch::optim::SGDOptions(config.lr)
                    .momentum(config.momentum)
                    .weight_decay(config.weight_decay)
                    .nesterov(config.nesterov);
    return std::make_unique<torch::optim::SGD>(parameters, opts);
  }
  case OptimizerType::Adam: {
    auto opts = torch::optim::AdamOptions(config.lr)
                    .betas(std::make_tuple(config.beta1, config.beta2))
                    .eps(config.eps)
                    .weight_decay(config.weight_decay);
    return std::make_unique<torch::optim::Adam>(parameters, opts);
  }
  case OptimizerType::AdamW: {
    auto opts = torch::optim::AdamWOptions(config.lr)
                    .betas(std::make_tuple(config.beta1, config.beta2))
                    .eps(config.eps)
                    .weight_decay(config.weight_decay);
    return std::make_unique<torch::optim::AdamW>(parameters, opts);
  }
  default:
    TORCH_CHECK(false, "Unknown optimizer type");
  }
}

at::Tensor bce_with_logits_sparse(const SparseTensor &logits,
                                  const at::Tensor &targets,
                                  double pos_weight) {
  auto pw = torch::tensor({pos_weight}, logits.F().options());
  auto loss_fn = torch::nn::BCEWithLogitsLoss(
      torch::nn::BCEWithLogitsLossOptions().pos_weight(pw));
  return loss_fn(logits.F(), targets);
}

at::Tensor cross_entropy_sparse(const SparseTensor &logits,
                                const at::Tensor &targets,
                                c10::optional<at::Tensor> weight) {
  if (weight.has_value()) {
    auto loss_fn = torch::nn::CrossEntropyLoss(
        torch::nn::CrossEntropyLossOptions().weight(*weight));
    return loss_fn(logits.F(), targets);
  } else {
    auto loss_fn = torch::nn::CrossEntropyLoss();
    return loss_fn(logits.F(), targets);
  }
}

SparseBatch sparse_collate_samples(const std::vector<SparseSample> &samples) {
  SparseBatch batch;
  batch.batch_size = static_cast<int64_t>(samples.size());

  if (samples.empty()) {
    return batch;
  }

  std::vector<at::Tensor> coords_list;
  std::vector<at::Tensor> feats_list;
  std::vector<at::Tensor> labels_list;
  bool has_labels = false;

  for (int64_t b = 0; b < static_cast<int64_t>(samples.size()); b++) {
    auto &s = samples[b];
    int64_t N = s.coordinates.size(0);

    auto batch_col =
        torch::full({N, 1}, b, torch::TensorOptions().dtype(torch::kInt));
    auto coords_with_batch = torch::cat({batch_col, s.coordinates}, 1);
    coords_list.push_back(coords_with_batch);
    feats_list.push_back(s.features);

    if (s.labels.defined() && s.labels.numel() > 0) {
      has_labels = true;
      labels_list.push_back(s.labels);
    }
  }

  batch.coordinates = torch::cat(coords_list, 0);
  batch.features = torch::cat(feats_list, 0);
  if (has_labels) {
    batch.labels = torch::cat(labels_list, 0);
  }

  return batch;
}

SparseDataset::SparseDataset(std::vector<SparseSample> samples)
    : samples_(std::move(samples)) {}

void SparseDataset::add_sample(SparseSample sample) {
  samples_.push_back(std::move(sample));
}

SparseSample SparseDataset::get(size_t index) {
  TORCH_CHECK(index < samples_.size(), "Index ", index, " out of range for dataset of size ", samples_.size());
  auto &s = samples_[index];

  return SparseSample(s.coordinates.clone(), s.features.clone(),
                      s.labels.defined() ? s.labels.clone() : at::Tensor());
}

torch::optional<size_t> SparseDataset::size() const {
  return samples_.size();
}

StepResult train_step(
    torch::nn::AnyModule &model, const SparseBatch &batch,
    std::function<at::Tensor(const at::Tensor &, const at::Tensor &)> criterion,
    torch::optim::Optimizer &optimizer,
    std::shared_ptr<CoordinateManager> manager) {

  auto stensor = SparseTensor(batch.features, batch.coordinates, manager);
  auto output = model.forward<SparseTensor>(stensor);
  auto loss = criterion(output.F(), batch.labels);

  optimizer.zero_grad();
  loss.backward();
  optimizer.step();

  StepResult result;
  result.loss = loss.item<double>();
  result.num_points = batch.features.size(0);
  return result;
}

EpochResult train_epoch(
    torch::nn::AnyModule &model, const std::vector<SparseBatch> &batches,
    std::function<at::Tensor(const at::Tensor &, const at::Tensor &)> criterion,
    torch::optim::Optimizer &optimizer, int dimension) {

  EpochResult result;

  for (auto &batch : batches) {
    auto mgr = std::make_shared<CoordinateManager>(
        dimension, CoordinateMapBackend::CPU);
    auto step = train_step(model, batch, criterion, optimizer, mgr);
    result.avg_loss += step.loss;
    result.total_points += step.num_points;
    result.num_batches++;
  }

  if (result.num_batches > 0) {
    result.avg_loss /= static_cast<double>(result.num_batches);
  }
  return result;
}

EpochResult evaluate(
    torch::nn::AnyModule &model, const std::vector<SparseBatch> &batches,
    std::function<at::Tensor(const at::Tensor &, const at::Tensor &)> criterion,
    int dimension) {

  torch::NoGradGuard no_grad;
  EpochResult result;

  for (auto &batch : batches) {
    auto mgr = std::make_shared<CoordinateManager>(
        dimension, CoordinateMapBackend::CPU);
    auto stensor = SparseTensor(batch.features, batch.coordinates, mgr);
    auto output = model.forward<SparseTensor>(stensor);
    auto loss = criterion(output.F(), batch.labels);

    result.avg_loss += loss.item<double>();
    result.total_points += batch.features.size(0);
    result.num_batches++;
  }

  if (result.num_batches > 0) {
    result.avg_loss /= static_cast<double>(result.num_batches);
  }
  return result;
}

GradientAccumulator::GradientAccumulator(torch::optim::Optimizer &optimizer,
                                         int accumulation_steps)
    : optimizer_(optimizer), accumulation_steps_(accumulation_steps) {
  TORCH_CHECK(accumulation_steps > 0,
              "accumulation_steps must be > 0, got ", accumulation_steps);
}

bool GradientAccumulator::backward_and_maybe_step(at::Tensor loss) {

  auto scaled_loss = loss / static_cast<double>(accumulation_steps_);
  scaled_loss.backward();
  current_step_++;

  if (current_step_ >= accumulation_steps_) {
    optimizer_.step();
    optimizer_.zero_grad();
    current_step_ = 0;
    return true;
  }
  return false;
}

void GradientAccumulator::flush() {
  if (current_step_ > 0) {
    optimizer_.step();
    optimizer_.zero_grad();
    current_step_ = 0;
  }
}

void GradientAccumulator::reset() { current_step_ = 0; }

StepLRScheduler::StepLRScheduler(torch::optim::Optimizer &optimizer,
                                 int step_size, double gamma)
    : optimizer_(optimizer), step_size_(step_size), gamma_(gamma) {

  if (!optimizer_.param_groups().empty()) {
    initial_lr_ = optimizer_.param_groups()[0].options().get_lr();
  } else {
    initial_lr_ = 1e-3;
  }
}

void StepLRScheduler::step() {
  epoch_++;
  if (epoch_ % step_size_ == 0) {
    double new_lr = current_lr() * gamma_;
    for (auto &group : optimizer_.param_groups()) {
      static_cast<torch::optim::OptimizerOptions &>(group.options())
          .set_lr(new_lr);
    }
  }
}

double StepLRScheduler::current_lr() const {
  if (!optimizer_.param_groups().empty()) {
    return optimizer_.param_groups()[0].options().get_lr();
  }
  return initial_lr_;
}

} 
} 
