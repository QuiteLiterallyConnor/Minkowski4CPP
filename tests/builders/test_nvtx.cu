#if __has_include(<nvtx3/nvtx3.hpp>)
#include <nvtx3/nvtx3.hpp>
#endif
#include <cudf/detail/nvtx/ranges.hpp>

void profile_function() {
  CUDF_FUNC_RANGE();
  cudf::thread_range range{"builder NVTX compatibility test"};
}
