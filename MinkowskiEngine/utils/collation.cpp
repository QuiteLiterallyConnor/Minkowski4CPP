
#include "collation.hpp"
#include <numeric>
#include <stdexcept>

namespace minkowski {

at::Tensor batched_coordinates(
    const std::vector<at::Tensor>& coords,
    c10::ScalarType dtype,
    c10::optional<torch::Device> device) {

  TORCH_CHECK(!coords.empty(), "Coordinates sequence must not be empty.");

  int64_t D = coords[0].size(1);
  for (size_t i = 0; i < coords.size(); ++i) {
    TORCH_CHECK(coords[i].dim() == 2,
                "All coordinates must be 2D arrays. Tensor ", i,
                " has ", coords[i].dim(), " dimensions.");
    TORCH_CHECK(coords[i].size(1) == D,
                "Dimension mismatch. Tensor 0 has ", D,
                " columns, tensor ", i, " has ", coords[i].size(1));
  }

  TORCH_CHECK(dtype == torch::kInt32 || dtype == torch::kFloat32,
              "Only torch::kInt32 and torch::kFloat32 are supported for coordinates.");

  torch::Device target_device = torch::kCPU;
  if (device.has_value()) {
    target_device = device.value();
  } else if (coords[0].defined()) {
    target_device = coords[0].device();
  }

  int64_t N = 0;
  for (const auto& cs : coords) {
    N += cs.size(0);
  }

  auto bcoords = torch::zeros({N, D + 1}, torch::TensorOptions().dtype(dtype).device(target_device));

  int64_t s = 0;
  for (int64_t b = 0; b < static_cast<int64_t>(coords.size()); ++b) {
    at::Tensor cs = coords[b];

    if (dtype == torch::kInt32) {
      if (cs.dtype() == torch::kFloat32 || cs.dtype() == torch::kFloat64) {
        cs = cs.floor();
      }
      cs = cs.to(torch::kInt32);
    } else {
      cs = cs.to(dtype);
    }

    if (cs.device() != target_device) {
      cs = cs.to(target_device);
    }

    int64_t cn = cs.size(0);

    bcoords.slice(0, s, s + cn).slice(1, 1) = cs;
    bcoords.slice(0, s, s + cn).select(1, 0).fill_(b);
    s += cn;
  }

  return bcoords;
}

std::tuple<at::Tensor, at::Tensor, c10::optional<at::Tensor>>
sparse_collate(
    const std::vector<at::Tensor>& coords,
    const std::vector<at::Tensor>& feats,
    const std::vector<at::Tensor>& labels,
    c10::ScalarType dtype,
    c10::optional<torch::Device> device) {

  TORCH_CHECK(!coords.empty(), "Coordinates must not be empty.");
  TORCH_CHECK(!feats.empty(), "Features must not be empty.");
  TORCH_CHECK(coords.size() == feats.size(),
              "Number of coordinate tensors (", coords.size(),
              ") must match number of feature tensors (", feats.size(), ").");

  bool use_labels = !labels.empty();
  if (use_labels) {
    TORCH_CHECK(labels.size() == coords.size(),
                "Number of label tensors must match number of coordinate tensors.");
  }

  int64_t D = coords[0].size(1);
  for (size_t i = 0; i < coords.size(); ++i) {
    TORCH_CHECK(coords[i].dim() == 2, "Coordinates must be 2D matrices.");
    TORCH_CHECK(coords[i].size(1) == D, "Coordinate dimension mismatch.");
    TORCH_CHECK(coords[i].size(0) == feats[i].size(0),
                "Coordinate count must match feature count for sample ", i);
  }

  TORCH_CHECK(dtype == torch::kInt32 || dtype == torch::kFloat32,
              "Only torch::kInt32 and torch::kFloat32 supported for coordinates.");

  torch::Device target_device = torch::kCPU;
  if (device.has_value()) {
    target_device = device.value();
  } else if (coords[0].defined()) {
    target_device = coords[0].device();
  }

  int64_t N = 0;
  for (const auto& cs : coords) N += cs.size(0);

  auto bcoords = torch::zeros({N, D + 1}, torch::TensorOptions().dtype(dtype).device(target_device));

  std::vector<at::Tensor> feats_batch;
  std::vector<at::Tensor> labels_batch;
  feats_batch.reserve(coords.size());
  if (use_labels) labels_batch.reserve(coords.size());

  int64_t s = 0;
  for (size_t b = 0; b < coords.size(); ++b) {
    at::Tensor coord = coords[b];
    at::Tensor feat = feats[b];

    if (dtype == torch::kInt32 &&
        (coord.dtype() == torch::kFloat32 || coord.dtype() == torch::kFloat64)) {
      coord = coord.floor();
    }
    coord = coord.to(dtype);

    feat = feat.to(torch::kFloat);

    int64_t cn = coord.size(0);
    bcoords.slice(0, s, s + cn).slice(1, 1) = coord.to(target_device);
    bcoords.slice(0, s, s + cn).select(1, 0).fill_(static_cast<int64_t>(b));

    feats_batch.push_back(feat);

    if (use_labels) {
      labels_batch.push_back(labels[b]);
    }

    s += cn;
  }

  auto feats_cat = torch::cat(feats_batch, 0);

  if (use_labels) {

    auto labels_cat = torch::cat(labels_batch, 0);
    return {bcoords, feats_cat, labels_cat};
  }

  return {bcoords, feats_cat, c10::nullopt};
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
batch_sparse_collate(
    const std::vector<std::tuple<at::Tensor, at::Tensor, at::Tensor>>& data,
    c10::ScalarType dtype,
    c10::optional<torch::Device> device) {

  std::vector<at::Tensor> coords, feats, labels;
  coords.reserve(data.size());
  feats.reserve(data.size());
  labels.reserve(data.size());

  for (const auto& [c, f, l] : data) {
    coords.push_back(c);
    feats.push_back(f);
    labels.push_back(l);
  }

  auto [bcoords, bfeats, blabels] = sparse_collate(coords, feats, labels, dtype, device);
  TORCH_CHECK(blabels.has_value(), "Labels should be present in batch_sparse_collate");
  return {bcoords, bfeats, blabels.value()};
}

SparseCollation::SparseCollation(int64_t limit_numpoints,
                                 c10::ScalarType dtype,
                                 c10::optional<torch::Device> device)
    : limit_numpoints_(limit_numpoints), dtype_(dtype), device_(device) {}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
SparseCollation::operator()(
    const std::vector<std::tuple<at::Tensor, at::Tensor, at::Tensor>>& list_data) const {

  std::vector<at::Tensor> coords_batch, feats_batch, labels_batch;

  int64_t batch_num_points = 0;
  for (size_t i = 0; i < list_data.size(); ++i) {
    const auto& [coord, feat, label] = list_data[i];
    int64_t num_points = coord.size(0);
    batch_num_points += num_points;

    if (limit_numpoints_ > 0 && batch_num_points > limit_numpoints_) {

      break;
    }

    coords_batch.push_back(coord);
    feats_batch.push_back(feat);
    labels_batch.push_back(label);
  }

  auto [bcoords, bfeats, blabels] = sparse_collate(
      coords_batch, feats_batch, labels_batch, dtype_, device_);
  TORCH_CHECK(blabels.has_value(), "Labels expected in SparseCollation");
  return {bcoords, bfeats, blabels.value()};
}

} 
