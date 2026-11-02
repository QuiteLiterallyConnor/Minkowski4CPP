#include "MinkowskiEngine/adaptive/part_split.hpp"
#include "part_split_reference.hpp"
#include "utils/test_framework.hpp"
#include <random>

namespace {
namespace split = minkowski::adaptive::detail;
using split_reference::Key;

void check(std::vector<Key> keys, int radius) {
  std::sort(keys.begin(), keys.end(), split_reference::less);
  const auto reference = split_reference::starts(keys, radius);
  const auto got = split::part_starts(keys, radius);
  CHECK(got == reference);
  const auto leaves = split::part_leaves(got, keys);
  CHECK(leaves == split_reference::leaves(reference, keys));
  size_t total = 0;
  for (const auto &part : leaves) {
    CHECK(std::is_sorted(part.begin(), part.end()));
    total += part.size();
  }
  CHECK_EQ(total, keys.size());
}
}

TEST(part_split_empty_and_singleton) {
  for (int radius : {0, 1, 2, 17}) {
    check({}, radius);
    check({Key{{-17, 0, 0}, 3}}, radius);
  }
}

TEST(part_split_level_resets_and_negative_floor) {
  std::vector<Key> keys{
      {{-10, 0, 0}, 0}, {{10, 0, 0}, 0},
      {{-41, 0, 0}, 2}, {{-40, 1, 0}, 2}, {{39, 0, 0}, 2}, {{40, 1, 0}, 2},
      {{-81, 0, 0}, 3}, {{80, 0, 1}, 3}};
  for (int radius : {0, 1, 2, 20}) check(keys, radius);
}

TEST(part_split_kernel_radius_boundary) {
  for (int radius : {0, 1, 2, 7}) {
    std::vector<Key> keys;
    for (int x : {-20, -19, -1, 0, 1, 19, 20}) {
      keys.push_back({{x, 0, 0}, 0});
      keys.push_back({{int64_t(x + radius) * 8, 1, 0}, 3});
      keys.push_back({{int64_t(x + radius + 1) * 8, 2, 0}, 3});
    }
    check(keys, radius);
  }
}

TEST(part_split_repeated_cells_and_single_level) {
  std::vector<Key> keys;
  for (int x = -16; x < 16; ++x)
    for (int y = 0; y < 100; ++y) keys.push_back({{x, y, 0}, 3});
  for (int radius : {0, 1, 5}) check(keys, radius);
}

TEST(part_split_extreme_coordinates_and_sparse_levels) {
  const int64_t edge = (int64_t(1) << 52) - 1;
  std::vector<Key> keys;
  for (uint32_t level : {0u, 2u, 17u, 30u})
    for (int64_t x : {-edge, -edge + 1, int64_t(-1), int64_t(0), edge - 1, edge})
      keys.push_back({{x, 0, 0}, level});
  for (int radius : {0, 1, 100}) check(keys, radius);
}

TEST(part_split_randomized_differential) {
  std::mt19937_64 random(20261102);
  for (int trial = 0; trial < 250; ++trial) {
    std::vector<Key> keys;
    const size_t n = 1 + random() % 4000;
    for (size_t i = 0; i < n; ++i) {
      const uint32_t level = uint32_t(random() % 12);
      const int64_t x = int64_t(random() % 101) - 50;
      const int64_t scale = int64_t(1) << level;
      keys.push_back({{x * scale + int64_t(random() % scale), int64_t(i), 0}, level});
    }
    check(std::move(keys), int(random() % 8));
  }
}

int main(int argc, char **argv) { return mink_test::run_all(argc, argv); }
