#include "MinkowskiEngine/utils/sphere.hpp"
#include "third_party/json.hpp"

#include <exception>
#include <iostream>

using json = nlohmann::json;

int main() {
  constexpr double radius = 100.0;
  constexpr double target_accuracy = 99.0;
  constexpr double center_x = 0.0;
  constexpr double center_y = 0.0;
  constexpr double center_z = 0.0;

  try {
    double achieved_accuracy = 0.0;
    const auto sphere = minkowski::adaptive::voxelize_sphere(
        radius, target_accuracy, {center_x, center_y, center_z}, 10'000'000, 10,
        &achieved_accuracy);

    const auto coordinates = sphere.coordinates().contiguous();
    const auto levels = sphere.levels().contiguous();
    const auto centers = sphere.centers().contiguous();
    const auto sizes = sphere.voxel_sizes().contiguous();
    const auto bounds = sphere.bounds();
    const auto bounds_min = bounds.first.contiguous();
    const auto bounds_max = bounds.second.contiguous();
    const auto features = sphere.features().to(torch::kCPU).contiguous();
    const auto coordinate_data = coordinates.accessor<int64_t, 2>();
    const auto level_data = levels.accessor<int32_t, 1>();
    const auto center_data = centers.accessor<double, 2>();
    const auto size_data = sizes.accessor<double, 2>();
    const auto bounds_min_data = bounds_min.accessor<double, 2>();
    const auto bounds_max_data = bounds_max.accessor<double, 2>();
    const auto feature_data = features.accessor<float, 2>();

    json output = {
        {"radius", radius},
        {"target_accuracy_percent", target_accuracy},
        {"achieved_accuracy_percent", achieved_accuracy},
        {"sphere_center", {center_x, center_y, center_z}},
        {"voxels", json::array()},
    };

    for (int64_t i = 0; i < sphere.size(); ++i) {
      output["voxels"].push_back({
          {"coordinates", {coordinate_data[i][0], coordinate_data[i][1],
                           coordinate_data[i][2]}},
          {"level", level_data[i]},
          {"bounds_min", {bounds_min_data[i][0], bounds_min_data[i][1],
                          bounds_min_data[i][2]}},
          {"bounds_max_exclusive", {bounds_max_data[i][0], bounds_max_data[i][1],
                                    bounds_max_data[i][2]}},
          {"center", {center_data[i][0], center_data[i][1], center_data[i][2]}},
          {"size", {size_data[i][0], size_data[i][1], size_data[i][2]}},
          {"occupied", feature_data[i][0] > 0.5f},
      });
    }

    std::cout << output.dump(2) << '\n';
  } catch (const std::exception &error) {
    std::cerr << "sphere: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
