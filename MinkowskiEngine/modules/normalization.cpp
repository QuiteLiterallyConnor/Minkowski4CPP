#include "normalization.hpp"
#include "pooling.hpp"
#include "broadcast.hpp"
namespace minkowski {
MinkowskiBatchNormImpl::MinkowskiBatchNormImpl(int num_features, double eps,
                                               double momentum, bool affine,
                                               bool track_running_stats) {
  bn_ = register_module(
      "bn", torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(num_features)
                                       .eps(eps)
                                       .momentum(momentum)
                                       .affine(affine)
                                       .track_running_stats(track_running_stats)));
}
void MinkowskiBatchNormImpl::reset() {
  bn_->reset();
}
SparseTensor
MinkowskiBatchNormImpl::forward(const SparseTensor &input) {
  auto output = bn_->forward(input.F());
  return SparseTensor(output, input.coordinate_map_key(),
                      input.coordinate_manager());
}
MinkowskiSyncBatchNormImpl::MinkowskiSyncBatchNormImpl(
    int num_features, double eps, double momentum, bool affine,
    bool track_running_stats) {
  bn_ = register_module(
      "bn", torch::nn::BatchNorm1d(torch::nn::BatchNorm1dOptions(num_features)
                                       .eps(eps)
                                       .momentum(momentum)
                                       .affine(affine)
                                       .track_running_stats(track_running_stats)));
}
void MinkowskiSyncBatchNormImpl::reset() {
  bn_->reset();
}
SparseTensor
MinkowskiSyncBatchNormImpl::forward(const SparseTensor &input) {
  auto output = bn_->forward(input.F());
  return SparseTensor(output, input.coordinate_map_key(),
                      input.coordinate_manager());
}
MinkowskiInstanceNormImpl::MinkowskiInstanceNormImpl(int num_features)
    : num_features_(num_features) {
  weight_ =
      register_parameter("weight", torch::ones({1, num_features}));
  bias_ =
      register_parameter("bias", torch::zeros({1, num_features}));
}
void MinkowskiInstanceNormImpl::reset() {
  torch::NoGradGuard no_grad;
  weight_.fill_(1);
  bias_.fill_(0);
}
SparseTensor
MinkowskiInstanceNormImpl::forward(const SparseTensor &input) {
  TORCH_CHECK(input.F().size(1) == num_features_,
              "Feature dimension mismatch: expected ", num_features_, " got ",
              input.F().size(1));
  auto manager = input.coordinate_manager();
  auto in_key =
      std::make_shared<CoordinateMapKey>(input.coordinate_map_key());
  auto glob_key = std::make_shared<CoordinateMapKey>(
      input.coordinate_map_key().get_coordinate_size());
  auto norm_feat = InstanceNormFunction::apply(
      input.F(), in_key, glob_key, manager);
  auto output = norm_feat * weight_ + bias_;
  return SparseTensor(output, input.coordinate_map_key(), manager);
}
MinkowskiStableInstanceNormImpl::MinkowskiStableInstanceNormImpl(
    int num_features)
    : num_features_(num_features) {
  weight_ =
      register_parameter("weight", torch::ones({1, num_features}));
  bias_ =
      register_parameter("bias", torch::zeros({1, num_features}));
  mean_in_ = std::make_shared<MinkowskiGlobalPoolingImpl>(
      PoolingMode::GLOBAL_AVG_POOLING_PYTORCH_INDEX);
  register_module("mean_in", mean_in_);
  glob_sum_ = std::make_shared<MinkowskiBroadcastAdditionImpl>();
  register_module("glob_sum", glob_sum_);
  glob_sum2_ = std::make_shared<MinkowskiBroadcastAdditionImpl>();
  register_module("glob_sum2", glob_sum2_);
  glob_mean_ = std::make_shared<MinkowskiGlobalPoolingImpl>(
      PoolingMode::GLOBAL_AVG_POOLING_PYTORCH_INDEX);
  register_module("glob_mean", glob_mean_);
  glob_times_ = std::make_shared<MinkowskiBroadcastMultiplicationImpl>();
  register_module("glob_times", glob_times_);
}
void MinkowskiStableInstanceNormImpl::reset() {
  torch::NoGradGuard no_grad;
  weight_.fill_(1);
  bias_.fill_(0);
}
SparseTensor
MinkowskiStableInstanceNormImpl::forward(const SparseTensor &input) {
  auto manager = input.coordinate_manager();
  auto neg_input =
      SparseTensor(-input.F(), input.coordinate_map_key(), manager);
  auto neg_mean = mean_in_->forward(neg_input);
  auto centered = glob_sum_->forward(input, neg_mean);
  auto centered_sq = SparseTensor(centered.F().pow(2),
                                   centered.coordinate_map_key(),
                                   centered.coordinate_manager());
  auto var_in = glob_mean_->forward(centered_sq);
  auto instd_feat = 1.0 / (var_in.F() + eps_).sqrt();
  auto instd = SparseTensor(instd_feat, var_in.coordinate_map_key(),
                              var_in.coordinate_manager());
  auto centered2 = glob_sum2_->forward(input, neg_mean);
  auto normed = glob_times_->forward(centered2, instd);
  auto output = normed.F() * weight_ + bias_;
  return SparseTensor(output, normed.coordinate_map_key(),
                      normed.coordinate_manager());
}
} 
