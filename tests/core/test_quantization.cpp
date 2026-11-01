// sparse_quantize must quantize like MinkowskiEngine (floor(coordinates / quantization_size) in
// float64) and keep integer coordinates exact.
#include "core_test_utils.hpp"

using namespace minkowski;

namespace {

int64_t unique_rows(const SparseQuantizeResult& r) { return r.coordinates.size(0); }

}  // namespace

TEST(quantize_integer_coordinates_stay_exact_beyond_float32) {
    // 2^24 and 2^24 + 1 are distinct int32 coordinates but the same float32 value.
    auto coords = torch::tensor({{0, 16777216, 0, 0}, {0, 16777217, 0, 0}}, torch::kInt32);
    auto r = sparse_quantize(coords);
    CHECK_EQ(unique_rows(r), 2);
    auto x = r.coordinates.select(1, 1).sort();
    CHECK_EQ(std::get<0>(x)[0].item<int64_t>(), 16777216);
    CHECK_EQ(std::get<0>(x)[1].item<int64_t>(), 16777217);
}

TEST(quantize_float64_coordinates_use_float64_math) {
    // 16777217.5 rounds to 16777218 in float32 (floor -> 16777218); float64 floor gives 16777217.
    auto coords = torch::tensor({{0.0, 16777217.5, 3.25, -0.5}}, torch::kFloat64);
    auto r = sparse_quantize(coords);
    CHECK_EQ(r.coordinates.scalar_type(), torch::kInt32);
    CHECK_EQ(r.coordinates[0][1].item<int64_t>(), 16777217);
    CHECK_EQ(r.coordinates[0][2].item<int64_t>(), 3);
    CHECK_EQ(r.coordinates[0][3].item<int64_t>(), -1);
}

TEST(quantize_quantization_size_matches_float64_floor) {
    // Coordinates near cell borders: float32 division moves some of them into the next cell.
    torch::manual_seed(71);
    auto xyz = torch::rand({4000, 3}, torch::kFloat64) * 200.0 - 100.0;
    auto coords = torch::cat({torch::zeros({4000, 1}, torch::kFloat64), xyz}, 1);
    const double qs = 0.05;
    auto r = sparse_quantize(coords, c10::nullopt, c10::nullopt, -100, qs);
    // Like MinkowskiEngine, every column is divided (the first column here is 0).
    auto expected = torch::floor(coords.index({r.unique_index}) / qs).to(torch::kInt64);
    CHECK(torch::equal(r.coordinates.to(torch::kInt64), expected));
    auto all = torch::floor(coords / qs).to(torch::kInt64);
    CHECK_EQ(unique_rows(r), std::get<0>(at::unique_dim(all, 0)).size(0));
}

TEST(quantize_features_follow_their_device) {
#ifdef CPU_ONLY
    throw mink_test::Skip("CPU_ONLY build");
#else
    if (!torch::cuda::is_available()) throw mink_test::Skip("CUDA device not available");
    // Quantize on the GPU while the features stay on the CPU.
    auto coords = torch::tensor({{0, 1, 2, 3}, {0, 1, 2, 3}, {0, 4, 5, 6}}, torch::kInt32);
    auto feats = torch::tensor({{1.0}, {2.0}, {3.0}});
    auto r = sparse_quantize(coords, feats, c10::nullopt, -100, c10::nullopt, torch::kCUDA);
    CHECK_EQ(r.coordinates.size(0), 2);
    CHECK(r.features.has_value());
    CHECK(r.features->device().is_cpu());
    CHECK_EQ(r.features->size(0), 2);
#endif
}
