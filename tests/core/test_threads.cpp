// Creating a coordinate manager (directly, through a SparseTensor or through sparse_quantize) must
// not change the process-wide intra-op / OpenMP thread count.
#ifdef _OPENMP
#include <omp.h>
#endif

#include "core_test_utils.hpp"

using namespace minkowski;

namespace {

struct ThreadCounts {
    int aten;
    int omp;
};

ThreadCounts thread_counts() {
#ifdef _OPENMP
    return {at::get_num_threads(), omp_get_max_threads()};
#else
    return {at::get_num_threads(), -1};
#endif
}

void check_unchanged(const ThreadCounts& before, const char* what) {
    const auto after = thread_counts();
    if (after.aten != before.aten || after.omp != before.omp)
        FAIL_AT(what << " changed the thread count: at::get_num_threads " << before.aten << " -> "
                     << after.aten << ", omp_get_max_threads " << before.omp << " -> "
                     << after.omp);
}

// Every check runs with a known multi-threaded setting, so the test is meaningful on any machine
// and restores the previous setting afterwards.
struct ThreadScope {
    int saved = at::get_num_threads();
    ThreadScope() { at::set_num_threads(2); }
    ~ThreadScope() { at::set_num_threads(saved); }
};

}  // namespace

TEST(threads_coordinate_manager_keeps_thread_count) {
    ThreadScope scope;
    const auto before = thread_counts();
    CHECK_EQ(before.aten, 2);
    { CoordinateManager manager(3, CoordinateMapBackend::CPU); }
    check_unchanged(before, "CoordinateManager(CPU)");
}

TEST(threads_sparse_tensor_keeps_thread_count) {
    ThreadScope scope;
    const auto before = thread_counts();
    auto coords = core_test::random_coords(50, 6, 2, 3, 61);
    SparseTensor x(torch::randn({coords.size(0), 2}), coords);
    MinkowskiConvolution conv(2, 2, 3, 2, 1, false, false, ConvolutionMode::DEFAULT, 3);
    auto y = conv->forward(x);
    y.F().sum().backward();
    check_unchanged(before, "SparseTensor + convolution");
}

TEST(threads_sparse_quantize_keeps_thread_count) {
    ThreadScope scope;
    const auto before = thread_counts();
    torch::manual_seed(62);
    auto coords = torch::cat({torch::zeros({200, 1}), torch::rand({200, 3}) * 10}, 1);  // b x y z
    auto result = sparse_quantize(coords);
    (void)result;
    check_unchanged(before, "sparse_quantize");
}
