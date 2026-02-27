
#ifndef MINK_CPP_TRAINING_TRAINING_HPP
#define MINK_CPP_TRAINING_TRAINING_HPP

#include "../sparse_tensor.hpp"
#include "../coordinate_manager.hpp"
#include "../utils/collation.hpp"
#include <torch/torch.h>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace minkowski {
namespace training {

enum class OptimizerType { SGD, Adam, AdamW };

struct OptimizerConfig {
  OptimizerType type = OptimizerType::AdamW;
  double lr = 1e-3;
  double weight_decay = 1e-4;
  double momentum = 0.9;           
  double beta1 = 0.9;              
  double beta2 = 0.999;            
  double eps = 1e-8;               
  bool nesterov = false;           
};

std::unique_ptr<torch::optim::Optimizer>
create_optimizer(const std::vector<torch::Tensor> &parameters,
                 const OptimizerConfig &config);

template <typename LossModule>
at::Tensor sparse_loss(const SparseTensor &output, const at::Tensor &target,
                       LossModule &criterion) {
  return criterion(output.F(), target);
}

at::Tensor bce_with_logits_sparse(const SparseTensor &logits,
                                  const at::Tensor &targets,
                                  double pos_weight = 1.0);

at::Tensor cross_entropy_sparse(
    const SparseTensor &logits, const at::Tensor &targets,
    c10::optional<at::Tensor> weight = c10::nullopt);

struct SparseSample {
  at::Tensor coordinates; 
  at::Tensor features;    
  at::Tensor labels;      

  SparseSample() = default;
  SparseSample(at::Tensor coords, at::Tensor feats, at::Tensor labs)
      : coordinates(std::move(coords)), features(std::move(feats)),
        labels(std::move(labs)) {}
  SparseSample(at::Tensor coords, at::Tensor feats)
      : coordinates(std::move(coords)), features(std::move(feats)) {}
};

struct SparseBatch {
  at::Tensor coordinates; 
  at::Tensor features;    
  at::Tensor labels;      
  int64_t batch_size;

  SparseBatch() : batch_size(0) {}
};

SparseBatch sparse_collate_samples(const std::vector<SparseSample> &samples);

class SparseDataset : public torch::data::datasets::Dataset<SparseDataset, SparseSample> {
public:
  SparseDataset() = default;

  explicit SparseDataset(std::vector<SparseSample> samples);

  void add_sample(SparseSample sample);

  SparseSample get(size_t index) override;
  torch::optional<size_t> size() const override;

private:
  std::vector<SparseSample> samples_;
};

struct StepResult {
  double loss = 0.0;
  int64_t num_points = 0;
};

struct EpochResult {
  double avg_loss = 0.0;
  int64_t total_points = 0;
  int64_t num_batches = 0;
};

StepResult train_step(
    torch::nn::AnyModule &model, const SparseBatch &batch,
    std::function<at::Tensor(const at::Tensor &, const at::Tensor &)> criterion,
    torch::optim::Optimizer &optimizer,
    std::shared_ptr<CoordinateManager> manager);

EpochResult train_epoch(
    torch::nn::AnyModule &model, const std::vector<SparseBatch> &batches,
    std::function<at::Tensor(const at::Tensor &, const at::Tensor &)> criterion,
    torch::optim::Optimizer &optimizer, int dimension = 3);

EpochResult evaluate(
    torch::nn::AnyModule &model, const std::vector<SparseBatch> &batches,
    std::function<at::Tensor(const at::Tensor &, const at::Tensor &)> criterion,
    int dimension = 3);

class GradientAccumulator {
public:

  explicit GradientAccumulator(torch::optim::Optimizer &optimizer,
                               int accumulation_steps = 1);

  bool backward_and_maybe_step(at::Tensor loss);

  void flush();

  void reset();

  int accumulation_steps() const { return accumulation_steps_; }
  int current_step() const { return current_step_; }

private:
  torch::optim::Optimizer &optimizer_;
  int accumulation_steps_;
  int current_step_ = 0;
};

class StepLRScheduler {
public:
  StepLRScheduler(torch::optim::Optimizer &optimizer, int step_size,
                  double gamma = 0.1);

  void step();

  int current_epoch() const { return epoch_; }
  double current_lr() const;

private:
  torch::optim::Optimizer &optimizer_;
  int step_size_;
  double gamma_;
  int epoch_ = 0;
  double initial_lr_;
};

} 
} 

#endif 
