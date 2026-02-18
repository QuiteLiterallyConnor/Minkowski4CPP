#include "union_op.hpp"
namespace minkowski {
SparseTensor
MinkowskiUnionImpl::forward(const std::vector<SparseTensor> &inputs) {
  TORCH_CHECK(inputs.size() > 1,
              "MinkowskiUnion requires at least 2 inputs, got ",
              inputs.size());
  auto manager = inputs[0].coordinate_manager();
  for (size_t i = 1; i < inputs.size(); ++i) {
    TORCH_CHECK(manager.get() == inputs[i].coordinate_manager().get(),
                "All inputs must share the same CoordinateManager");
  }
  std::vector<CoordinateMapKey> in_keys;
  for (const auto &s : inputs) {
    in_keys.push_back(s.coordinate_map_key());
  }
  auto out_key = CoordinateMapKey(
      inputs[0].coordinate_map_key().get_coordinate_size());
  auto union_maps = manager->union_map(in_keys, out_key);
  int64_t out_nrows =
      static_cast<int64_t>(manager->size(out_key));
  int64_t n_channels = inputs[0].F().size(1);
  auto out_feat = torch::zeros({out_nrows, n_channels},
                                inputs[0].F().options());
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto in_map = union_maps[i].index({0}).to(torch::kLong);
    auto out_map = union_maps[i].index({1}).to(torch::kLong);
    out_feat.index_put_({out_map}, out_feat.index({out_map}) +
                                       inputs[i].F().index({in_map}));
  }
  return SparseTensor(out_feat, out_key, manager);
}
} 
