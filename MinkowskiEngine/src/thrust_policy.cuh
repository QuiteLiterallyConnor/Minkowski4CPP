#ifndef MINK_THRUST_POLICY_CUH
#define MINK_THRUST_POLICY_CUH

// Execution policy for thrust algorithms in the coordinate / kernel map code.
//
// `thrust::device` allocates the temporary storage of sort, scan, reduce_by_key, remove_if, ...
// with cudaMalloc and releases it with cudaFree, which synchronizes the whole device. These
// algorithms run several times per coordinate map and kernel map. This policy takes the
// temporaries from PyTorch's caching allocator instead. Work stays on the legacy default
// stream (as with `thrust::device`), and the blocks are allocated for that stream, so reuse
// after release is stream-ordered.

#include <cstddef>

#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>

namespace minkowski {
namespace detail {

struct c10_thrust_temporary_allocator {
  using value_type = char;

  char *allocate(std::ptrdiff_t num_bytes) {
    return static_cast<char *>(
        c10::cuda::CUDACachingAllocator::raw_alloc_with_stream(
            static_cast<size_t>(num_bytes), cudaStream_t(0)));
  }

  void deallocate(char *ptr, size_t) {
    c10::cuda::CUDACachingAllocator::raw_delete(ptr);
  }
};

inline auto thrust_policy() {
  static c10_thrust_temporary_allocator allocator; // stateless
  return thrust::cuda::par(allocator).on(cudaStream_t(0));
}

} // namespace detail
} // namespace minkowski

#endif // MINK_THRUST_POLICY_CUH
