#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "adaptive/adaptive.hpp"
#include "sphere_volume_cuda.hpp"

using minkowski::adaptive::AdaptiveSparseTensor;
using minkowski::adaptive::AdaptiveVoxelConfig;
using Clock = std::chrono::steady_clock;

namespace {

constexpr double kRadius = 100.0;
constexpr double kSide = 2.0 * kRadius;
constexpr double kPi = 3.14159265358979323846;
using Cell = std::array<int64_t, 4>; // x, y, z, level

// Below this a GPU round trip costs more than the host work it replaces.
constexpr int64_t kDeviceMinVoxels = 1 << 15;

enum class Relation { Outside, Inside, Boundary };

struct Estimate {
  double volume = 0.0;
  int64_t inside_voxels = 0;
  int64_t outside_voxels = 0;
  int64_t boundary_voxels = 0;
  uint32_t max_level = 0;
  std::vector<int64_t> counts_by_level;
};

double sphere_volume(double radius) {
  return (4.0 / 3.0) * kPi * radius * radius * radius;
}

AdaptiveVoxelConfig make_config(double base_size, uint32_t depth) {
  AdaptiveVoxelConfig config;
  config.base_voxel_size = {base_size, base_size, base_size};
  config.min_voxel_size = {std::ldexp(base_size, -static_cast<int>(depth)),
                           std::ldexp(base_size, -static_cast<int>(depth)),
                           std::ldexp(base_size, -static_cast<int>(depth))};
  config.max_depth = depth;
  return config;
}

Relation classify_cell(const Cell& cell, double base_size) {
  const auto level = static_cast<uint32_t>(cell[3]);
  const double size = std::ldexp(base_size, -static_cast<int>(level));
  double nearest_distance_squared = 0.0;
  double farthest_distance_squared = 0.0;

  for (int axis = 0; axis < 3; ++axis) {
    const double lo = static_cast<double>(cell[axis]) * size;
    const double hi = lo + size;
    const double nearest = std::max(lo, std::min(kRadius, hi));
    const double nearest_delta = nearest - kRadius;
    const double farthest_delta = std::max(std::abs(lo - kRadius), std::abs(hi - kRadius));
    nearest_distance_squared += nearest_delta * nearest_delta;
    farthest_distance_squared += farthest_delta * farthest_delta;
  }

  const double radius_squared = kRadius * kRadius;
  if (nearest_distance_squared >= radius_squared)
    return Relation::Outside;
  if (farthest_distance_squared <= radius_squared)
    return Relation::Inside;
  return Relation::Boundary;
}

// Children are enumerated with x as the slowest and z as the fastest axis, so a level's
// cells stay in lexicographic order when parents are processed in order.
inline Cell child_cell(const Cell& parent, int child) {
  return {2 * parent[0] + ((child >> 2) & 1), 2 * parent[1] + ((child >> 1) & 1),
          2 * parent[2] + (child & 1), parent[3] + 1};
}

std::vector<Cell> make_adaptive_cells(int64_t voxel_budget, uint32_t max_depth) {
  // Refine boundary cells one level at a time (largest cells first) until the budget
  // would be exceeded. Cells of one level are the same size, so this is equivalent to
  // always splitting the largest boundary cell, without rescanning all leaves.
  // Leaves are emitted level by level (not globally sorted; the engine sorts them).
  std::vector<Cell> leaves;
  std::vector<Cell> current{{0, 0, 0, 0}};
  std::vector<char> current_boundary{1};
  size_t count = 1;
  while (!current.empty()) {
    std::vector<Cell> next;
    std::vector<char> next_boundary;
    bool stop = false;
    for (size_t i = 0; i < current.size(); ++i) {
      if (current_boundary[i] && !stop &&
          (static_cast<uint32_t>(current[i][3]) >= max_depth ||
           count + 7 > static_cast<size_t>(voxel_budget)))
        stop = true;
      if (!current_boundary[i] || stop) {
        leaves.push_back(current[i]);
        continue;
      }
      count += 7;
      for (int child = 0; child < 8; ++child) {
        const Cell c = child_cell(current[i], child);
        next.push_back(c);
        next_boundary.push_back(classify_cell(c, kSide) == Relation::Boundary);
      }
    }
    if (stop) {
      leaves.insert(leaves.end(), next.begin(), next.end());
      return leaves;
    }
    current = std::move(next);
    current_boundary = std::move(next_boundary);
  }
  return leaves;
}

AdaptiveSparseTensor make_tensor(const std::vector<Cell>& cells, double base_size,
                                 uint32_t max_depth) {
  auto coordinates = torch::zeros({static_cast<int64_t>(cells.size()), 3}, torch::kInt64);
  auto levels = torch::zeros({static_cast<int64_t>(cells.size())}, torch::kInt32);
  auto features = torch::zeros({static_cast<int64_t>(cells.size()), 1}, torch::kFloat64);
  auto coordinate_data = coordinates.accessor<int64_t, 2>();
  auto level_data = levels.accessor<int32_t, 1>();
  for (size_t i = 0; i < cells.size(); ++i) {
    for (int axis = 0; axis < 3; ++axis)
      coordinate_data[static_cast<int64_t>(i)][axis] = cells[i][axis];
    level_data[static_cast<int64_t>(i)] = static_cast<int32_t>(cells[i][3]);
  }
  return AdaptiveSparseTensor::from_voxels(
      coordinates, levels, features, make_config(base_size, max_depth));
}

std::vector<Cell> make_uniform_cells(int cells_per_axis) {
  std::vector<Cell> cells;
  cells.reserve(static_cast<size_t>(cells_per_axis) * cells_per_axis * cells_per_axis);
  for (int64_t x = 0; x < cells_per_axis; ++x)
    for (int64_t y = 0; y < cells_per_axis; ++y)
      for (int64_t z = 0; z < cells_per_axis; ++z)
        cells.push_back({x, y, z, 0});
  return cells;
}

AdaptiveSparseTensor from_device_cells(std::vector<int64_t>& coords, std::vector<int32_t>& levels,
                                       double base_size, uint32_t max_depth) {
  const int64_t n = static_cast<int64_t>(levels.size());
  auto coordinates = torch::from_blob(coords.data(), {n, 3}, torch::kInt64);
  auto level_tensor = torch::from_blob(levels.data(), {n}, torch::kInt32);
  auto features = torch::zeros({n, 1}, torch::kFloat64);
  return AdaptiveSparseTensor::from_voxels(coordinates, level_tensor, features,
                                           make_config(base_size, max_depth));
}

// Cells are generated on the GPU when one is available, otherwise on the host.
AdaptiveSparseTensor build_adaptive(int64_t budget, uint32_t max_depth) {
  if (budget < kDeviceMinVoxels || !sphere_cuda::available())
    return make_tensor(make_adaptive_cells(budget, max_depth), kSide, max_depth);
  std::vector<int64_t> coords;
  std::vector<int32_t> levels;
  sphere_cuda::generate_adaptive(budget, max_depth, kSide, kRadius, coords, levels);
  return from_device_cells(coords, levels, kSide, max_depth);
}

AdaptiveSparseTensor build_uniform(int cells_per_axis, double edge) {
  if (int64_t(cells_per_axis) * cells_per_axis * cells_per_axis < kDeviceMinVoxels ||
      !sphere_cuda::available())
    return make_tensor(make_uniform_cells(cells_per_axis), edge, 0);
  std::vector<int64_t> coords;
  std::vector<int32_t> levels;
  sphere_cuda::generate_uniform(cells_per_axis, coords, levels);
  return from_device_cells(coords, levels, edge, 0);
}

Estimate estimate_volume(const AdaptiveSparseTensor& field) {
  if (field.size() >= kDeviceMinVoxels && sphere_cuda::available()) {
    const auto coordinates = field.coordinates().contiguous();
    const auto levels = field.levels().contiguous();
    const auto device = sphere_cuda::estimate(coordinates.data_ptr<int64_t>(),
                                              levels.data_ptr<int32_t>(), field.size(),
                                              field.config().base_voxel_size[0], kRadius);
    Estimate estimate;
    estimate.counts_by_level = field.counts_by_level();
    for (size_t level = 0; level < estimate.counts_by_level.size(); ++level)
      if (estimate.counts_by_level[level] > 0)
        estimate.max_level = static_cast<uint32_t>(level);
    estimate.volume = device.volume;
    estimate.inside_voxels = device.inside;
    estimate.outside_voxels = field.size() - device.inside;
    estimate.boundary_voxels = device.boundary;
    return estimate;
  }
  const auto centers = field.centers().contiguous();
  const auto volumes = field.volumes().contiguous();
  const double* center_data = centers.data_ptr<double>();
  const double* volume_data = volumes.data_ptr<double>();

  Estimate estimate;
  estimate.counts_by_level = field.counts_by_level();
  for (size_t level = 0; level < estimate.counts_by_level.size(); ++level)
    if (estimate.counts_by_level[level] > 0)
      estimate.max_level = static_cast<uint32_t>(level);

  for (int64_t i = 0; i < field.size(); ++i) {
    const double dx = center_data[i * 3] - kRadius;
    const double dy = center_data[i * 3 + 1] - kRadius;
    const double dz = center_data[i * 3 + 2] - kRadius;
    if (dx * dx + dy * dy + dz * dz <= kRadius * kRadius) {
      estimate.volume += volume_data[i];
      ++estimate.inside_voxels;
    } else {
      ++estimate.outside_voxels;
    }
  }

  const auto coordinates = field.coordinates().contiguous();
  const auto levels = field.levels().contiguous();
  const int64_t* coordinate_data = coordinates.data_ptr<int64_t>();
  const int32_t* level_data = levels.data_ptr<int32_t>();
  for (int64_t i = 0; i < field.size(); ++i) {
    const Cell cell{coordinate_data[i * 3], coordinate_data[i * 3 + 1],
                    coordinate_data[i * 3 + 2], level_data[i]};
    switch (classify_cell(cell, field.config().base_voxel_size[0])) {
      case Relation::Inside:
        break;
      case Relation::Outside:
        break;
      case Relation::Boundary:
        ++estimate.boundary_voxels;
        break;
    }
  }
  return estimate;
}

template <typename Function>
double median_ms(Function&& function, int repetitions = 7) {
  if (repetitions > 1)
    function(); // warm-up; skipped for very large grids to keep the run short
  std::vector<double> samples;
  samples.reserve(repetitions);
  for (int i = 0; i < repetitions; ++i) {
    const auto start = Clock::now();
    function();
    samples.push_back(std::chrono::duration<double, std::milli>(Clock::now() - start).count());
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

double accuracy_percent(double estimated_volume) {
  return 100.0 * (1.0 - std::abs(estimated_volume - sphere_volume(kRadius)) / sphere_volume(kRadius));
}

// Smallest cubic grid (cells per axis) whose accuracy reaches the target; 0 if none up to
// max_voxels. Accuracy is not monotonic in grid size, so this is the first size to reach it.
int find_uniform_axis(double target_percent, int64_t max_voxels, double& achieved) {
  const int max_axis = static_cast<int>(std::cbrt(static_cast<double>(max_voxels)));
  for (int n = 1; n <= max_axis; ++n) {
    const double edge = kSide / n;
    std::vector<double> squared(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      const double d = (i + 0.5) * edge - kRadius;
      squared[static_cast<size_t>(i)] = d * d;
    }
    int64_t inside = 0;
    for (int x = 0; x < n; ++x)
      for (int y = 0; y < n; ++y) {
        const double xy = squared[x] + squared[y];
        for (int z = 0; z < n; ++z)
          inside += xy + squared[z] <= kRadius * kRadius;
      }
    achieved = accuracy_percent(static_cast<double>(inside) * edge * edge * edge);
    if (achieved >= target_percent)
      return n;
  }
  return 0;
}

// Replays the adaptive refinement (the same order as make_adaptive_cells) one split at a
// time, tracking the estimate incrementally. Returns the smallest voxel count that reaches
// the target, or 0 if none up to max_voxels.
int64_t find_adaptive_budget(double target_percent, int64_t max_voxels, uint32_t max_depth,
                             double& achieved) {
  const auto contribution = [](const Cell& c) {
    const double size = std::ldexp(kSide, -static_cast<int>(c[3]));
    double d2 = 0.0;
    for (int axis = 0; axis < 3; ++axis) {
      const double d = (static_cast<double>(c[axis]) + 0.5) * size - kRadius;
      d2 += d * d;
    }
    return d2 <= kRadius * kRadius ? size * size * size : 0.0;
  };
  std::deque<Cell> queue;
  Cell root{0, 0, 0, 0};
  double volume = contribution(root);
  int64_t count = 1;
  if (classify_cell(root, kSide) == Relation::Boundary)
    queue.push_back(root);
  while (true) {
    achieved = accuracy_percent(volume);
    if (achieved >= target_percent)
      return count;
    if (queue.empty() || count + 7 > max_voxels ||
        static_cast<uint32_t>(queue.front()[3]) >= max_depth)
      return 0;
    const Cell parent = queue.front();
    queue.pop_front();
    volume -= contribution(parent);
    count += 7;
    for (int child = 0; child < 8; ++child) {
      const Cell c = child_cell(parent, child);
      volume += contribution(c);
      if (classify_cell(c, kSide) == Relation::Boundary)
        queue.push_back(c);
    }
  }
}

void print_result(const char* label, int64_t budget, const AdaptiveSparseTensor& field,
                  const Estimate& estimate, double build_ms, double estimate_ms,
                  double end_to_end_ms, double exact_volume) {
  const double signed_error = estimate.volume - exact_volume;
  const double relative_error = std::abs(signed_error) / exact_volume;
  std::cout << "\n" << label << " (target " << budget << ", actual " << field.size()
            << " voxels)\n"
            << "  estimated volume       " << estimate.volume << " m^3\n"
            << "  absolute error         " << std::abs(signed_error) << " m^3\n"
            << "  signed bias            " << signed_error << " m^3\n"
            << "  relative accuracy      " << 100.0 * (1.0 - relative_error) << "%\n"
            << "  center-inside voxels   " << estimate.inside_voxels << "\n"
            << "  center-outside voxels  " << estimate.outside_voxels << "\n"
            << "  boundary voxels        " << estimate.boundary_voxels << "\n"
            << "  max refinement level   " << estimate.max_level << "\n"
            << "  voxels by level        ";
  bool first = true;
  for (size_t level = 0; level < estimate.counts_by_level.size(); ++level) {
    if (estimate.counts_by_level[level] == 0)
      continue;
    std::cout << (first ? "" : ", ") << "L" << level << "="
              << estimate.counts_by_level[level];
    first = false;
  }
  std::cout << "\n"
            << "  topology build         " << build_ms << " ms\n"
            << "  volume evaluation      " << estimate_ms << " ms\n"
            << "  end-to-end             " << end_to_end_ms << " ms\n";
}

struct SweepPoint {
  std::string method;
  int64_t voxels;
  double ms;
  double accuracy;
  double digits; // -log10(relative error), capped
};

double digits_of(double accuracy) {
  const double err = std::max(1.0 - accuracy / 100.0, 1e-9);
  return -std::log10(err);
}

// Sweeps voxel counts for both methods, timing build + estimate end to end, then reports
// each method's fastest point, best digits-of-accuracy per ms, and Pareto frontier.
void run_optimal(uint32_t max_depth, int64_t max_voxels) {
  std::vector<SweepPoint> points;
  const auto measure = [&](const char* method, int64_t voxels, auto&& make_field,
                           int reps) {
    double accuracy = 0.0;
    int64_t actual = 0;
    const double ms = median_ms(
        [&] {
          auto field = make_field();
          const Estimate e = estimate_volume(field);
          accuracy = accuracy_percent(e.volume);
          actual = field.size();
        },
        reps);
    (void)voxels;
    points.push_back({method, actual, ms, accuracy, digits_of(accuracy)});
  };

  int64_t last = 0;
  for (double b = 1.0; b <= double(max_voxels); b *= 1.12) {
    const int64_t budget = static_cast<int64_t>(b);
    if (budget == last)
      continue;
    last = budget;
    measure("Adaptive", budget,
            [&] { return build_adaptive(budget, max_depth); },
            budget > 200000 ? 3 : 7);
  }
  const int max_axis = static_cast<int>(std::cbrt(static_cast<double>(max_voxels)));
  for (int n = 1; n <= max_axis; n += (n < 40 ? 1 : 3)) {
    measure("Uniform", int64_t(n) * n * n,
            [&] { return build_uniform(n, kSide / n); },
            n > 58 ? 3 : 7);
  }

  std::cout << "\nLatency vs accuracy sweep (end-to-end = build + volume evaluation)\n"
            << "digits = -log10(relative error), so 99.9% = 3 digits\n";
  for (const char* method : {"Adaptive", "Uniform"}) {
    const SweepPoint* fastest = nullptr;
    const SweepPoint* best_rate = nullptr;
    const SweepPoint* best_throughput = nullptr;
    std::vector<const SweepPoint*> mine;
    for (const auto& p : points)
      if (p.method == method)
        mine.push_back(&p);
    for (const auto* p : mine) {
      if (!fastest || p->ms < fastest->ms)
        fastest = p;
      if (!best_rate || p->digits / p->ms > best_rate->digits / best_rate->ms)
        best_rate = p;
      if (!best_throughput || p->voxels / p->ms > best_throughput->voxels / best_throughput->ms)
        best_throughput = p;
    }
    std::cout << "\n" << method << "\n";
    const auto show = [](const char* label, const SweepPoint* p) {
      std::cout << "  " << label << p->voxels << " voxels, " << p->ms << " ms, accuracy "
                << p->accuracy << "%, " << p->digits << " digits, " << p->digits / p->ms
                << " digits/ms, " << p->voxels / p->ms << " voxels/ms\n";
    };
    show("fastest point       ", fastest);
    show("best digits per ms  ", best_rate);
    show("best voxels per ms  ", best_throughput);
    std::cout << "  Pareto frontier (a point is kept if nothing faster is as accurate):\n";
    std::vector<const SweepPoint*> sorted = mine;
    std::sort(sorted.begin(), sorted.end(),
              [](auto* a, auto* b) { return a->ms < b->ms; });
    double best_digits = -1.0;
    for (const auto* p : sorted)
      if (p->digits > best_digits) {
        best_digits = p->digits;
        std::cout << "    " << p->ms << " ms  " << p->accuracy << "%  (" << p->voxels
                  << " voxels)\n";
      }
  }

  // Combined frontier across both methods: which one wins at each latency.
  std::vector<const SweepPoint*> all;
  for (const auto& p : points)
    all.push_back(&p);
  std::sort(all.begin(), all.end(), [](auto* a, auto* b) { return a->ms < b->ms; });
  const SweepPoint* overall_fast = all.front();
  const SweepPoint* overall_rate = all.front();
  const SweepPoint* overall_throughput = all.front();
  for (const auto* p : all) {
    if (p->digits / p->ms > overall_rate->digits / overall_rate->ms)
      overall_rate = p;
    if (p->voxels / p->ms > overall_throughput->voxels / overall_throughput->ms)
      overall_throughput = p;
  }
  std::cout << "\nOverall\n"
            << "  fastest absolute latency: " << overall_fast->method << " ("
            << overall_fast->voxels << " voxels, " << overall_fast->ms << " ms, "
            << overall_fast->accuracy << "%)\n"
            << "  best voxels per ms:       " << overall_throughput->method << " ("
            << overall_throughput->voxels << " voxels, " << overall_throughput->ms << " ms, "
            << overall_throughput->voxels / overall_throughput->ms << " voxels/ms)\n"
            << "  best accuracy per ms:     " << overall_rate->method << " ("
            << overall_rate->voxels << " voxels, " << overall_rate->ms << " ms, "
            << overall_rate->accuracy << "%, " << overall_rate->digits / overall_rate->ms
            << " digits/ms)\n"
            << "  combined Pareto frontier:\n";
  double best_digits = -1.0;
  for (const auto* p : all)
    if (p->digits > best_digits) {
      best_digits = p->digits;
      std::cout << "    " << p->method << "  " << p->ms << " ms  " << p->accuracy << "%  ("
                << p->voxels << " voxels)\n";
    }
}

} // namespace

int main(int argc, char** argv) {
  try {
    // Usage: sphere_volume_benchmark [total_voxels]            (default 1000)
    //        sphere_volume_benchmark --accuracy <percent>      (voxels needed to reach it)
    //        sphere_volume_benchmark --optimal                 (latency/accuracy sweep)
    constexpr int64_t kMaxVoxels = 10'000'000;
    constexpr uint32_t kMaxDepth = 10;
    if (argc == 2 && std::string(argv[1]) == "--optimal") {
      std::cout << std::fixed << std::setprecision(3);
      run_optimal(kMaxDepth, 3'000'000);
      return 0;
    }
    const auto usage = [&] {
      std::cerr << "usage: sphere_volume_benchmark [total_voxels, 1 to " << kMaxVoxels
                << "]\n       sphere_volume_benchmark --accuracy <percent, greater than 0 and "
                   "below 100>\n";
      return 2;
    };
    bool accuracy_mode = false;
    double target_accuracy = 0.0;
    int64_t target_voxels = 1000;
    if (argc > 1 && std::string(argv[1]) == "--accuracy") {
      char* end = nullptr;
      if (argc != 3)
        return usage();
      target_accuracy = std::strtod(argv[2], &end);
      if (end == argv[2] || *end != '\0' || !(target_accuracy > 0.0 && target_accuracy < 100.0))
        return usage();
      accuracy_mode = true;
    } else if (argc > 1) {
      char* end = nullptr;
      const double parsed = std::strtod(argv[1], &end);
      if (argc != 2 || end == argv[1] || *end != '\0' ||
          !(parsed >= 1.0 && parsed <= double(kMaxVoxels)))
        return usage();
      target_voxels = static_cast<int64_t>(std::llround(parsed));
    }

    const double exact_volume = sphere_volume(kRadius);
    volatile double radius_input = kRadius;
    volatile double volume_sink = 0.0;
    constexpr int kAnalyticRepetitions = 100000;
    const double analytic_us_per_query =
        median_ms([&] {
          double total = 0.0;
          for (int i = 0; i < kAnalyticRepetitions; ++i)
            total += sphere_volume(radius_input);
          volume_sink = total;
        }) *
        1000.0 / kAnalyticRepetitions;
    (void)volume_sink;
    std::cout << std::fixed << std::setprecision(3)
              << "Sphere volume benchmark (radius " << kRadius << " m)\n"
              << "Voxel estimate: include a voxel's full volume when its center is inside.\n"
              << "Analytic reference volume: " << exact_volume << " m^3\n"
              << "Unvoxelized analytic reference: 100.000% accuracy, 0 voxels\n"
              << "Analytic formula evaluation: " << analytic_us_per_query << " us/query\n";

    int cells_per_axis = 0;
    int64_t adaptive_count = 0;
    if (accuracy_mode) {
      double uniform_accuracy = 0.0, adaptive_accuracy = 0.0;
      cells_per_axis = find_uniform_axis(target_accuracy, kMaxVoxels, uniform_accuracy);
      adaptive_count =
          find_adaptive_budget(target_accuracy, kMaxVoxels, kMaxDepth, adaptive_accuracy);
      const int64_t uniform_count =
          static_cast<int64_t>(cells_per_axis) * cells_per_axis * cells_per_axis;
      std::cout << "\nVoxels required to reach " << target_accuracy << "% accuracy"
                << " (first size that reaches it, searched up to " << kMaxVoxels << ")\n";
      if (adaptive_count > 0)
        std::cout << "  Adaptive  " << adaptive_count << " voxels (" << adaptive_accuracy
                  << "%)\n";
      else
        std::cout << "  Adaptive  not reached within " << kMaxVoxels << " voxels\n";
      if (cells_per_axis > 0)
        std::cout << "  Uniform   " << uniform_count << " voxels = " << cells_per_axis
                  << "^3 (" << uniform_accuracy << "%)\n";
      else
        std::cout << "  Uniform   not reached within " << kMaxVoxels << " voxels\n";
      if (adaptive_count > 0 && cells_per_axis > 0)
        std::cout << "  Uniform needs " << double(uniform_count) / double(adaptive_count)
                  << "x the voxels of adaptive\n";
    } else {
      // A uniform grid must be a cube, so use the nearest cubic voxel count.
      cells_per_axis =
          std::max(1, static_cast<int>(std::llround(std::cbrt(double(target_voxels)))));
      adaptive_count = static_cast<int64_t>(cells_per_axis) * cells_per_axis * cells_per_axis;
      std::cout << "Requested " << target_voxels << " voxels -> uniform grid " << cells_per_axis
                << "^3 = " << adaptive_count << " voxels; the adaptive field gets the same budget\n";
    }

    const int64_t uniform_count =
        static_cast<int64_t>(cells_per_axis) * cells_per_axis * cells_per_axis;
    const double voxel_edge = cells_per_axis > 0 ? kSide / cells_per_axis : 0.0;

    if (adaptive_count > 0) {
      const int reps = adaptive_count > 200000 ? 1 : 7;
      auto adaptive_field = build_adaptive(adaptive_count, kMaxDepth);
      const Estimate adaptive_estimate = estimate_volume(adaptive_field);
      const double adaptive_build_ms = median_ms([&] {
        auto field = build_adaptive(adaptive_count, kMaxDepth);
        (void)field;
      }, reps);
      const double adaptive_estimate_ms =
          median_ms([&] { (void)estimate_volume(adaptive_field); }, reps);
      const double adaptive_total_ms = median_ms([&] {
        auto field = build_adaptive(adaptive_count, kMaxDepth);
        (void)estimate_volume(field);
      }, reps);
      print_result("Adaptive", adaptive_count, adaptive_field, adaptive_estimate, adaptive_build_ms,
                   adaptive_estimate_ms, adaptive_total_ms, exact_volume);
    }

    if (cells_per_axis > 0) {
      const int reps = uniform_count > 200000 ? 1 : 7;
      auto uniform_field = build_uniform(cells_per_axis, voxel_edge);
      const Estimate estimate = estimate_volume(uniform_field);
      const double build_ms = median_ms([&] {
        auto candidate = build_uniform(cells_per_axis, voxel_edge);
        (void)candidate;
      }, reps);
      const double estimate_ms = median_ms([&] { (void)estimate_volume(uniform_field); }, reps);
      const double total_ms = median_ms([&] {
        auto candidate = build_uniform(cells_per_axis, voxel_edge);
        (void)estimate_volume(candidate);
      }, reps);
      print_result("Uniform", uniform_count, uniform_field, estimate, build_ms, estimate_ms,
                   total_ms, exact_volume);
      std::cout << "  voxel edge length      " << voxel_edge << " m\n"
                << "  grid dimensions        " << cells_per_axis << " x " << cells_per_axis
                << " x " << cells_per_axis << "\n";
    }
  } catch (const std::exception& error) {
    std::cerr << "sphere_volume_benchmark failed: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
