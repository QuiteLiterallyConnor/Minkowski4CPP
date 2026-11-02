// Device composition of a kernel map from stored per-part maps (part reuse, kernel_map.cpp).
// The part maps' pair arrays stay on the device between batches; a new batch map uploads only
// a segment table and the parts' leaf numbering, and one kernel writes the batch's in/out
// index arrays (and weights) straight into device memory.
#include "kernel_map.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <algorithm>

namespace minkowski {
namespace adaptive {
namespace detail {
namespace {

// One block per segment (slot, level, part): pairs q0 .. q0 + n of part p go to dst .. dst + n.
// Part p's arrays are at ptrs[4p] (in), ptrs[4p + 1] (out), ptrs[4p + 2] (weight, double, 0 if
// unweighted); ptrs[4p + 3] is the index width in bytes (2: uint16, 4: int32). Its local leaf i
// is global leaf leaves_in[bases[2p] + i] (inputs) / leaves_out[bases[2p + 1] + i] (outputs).
__global__ void compose_parts_kernel(const int64_t *__restrict__ seg, int64_t segments,
                                     const int64_t *__restrict__ ptrs,
                                     const int64_t *__restrict__ bases,
                                     const int64_t *__restrict__ leaves_in,
                                     const int64_t *__restrict__ leaves_out,
                                     int64_t *__restrict__ in_index,
                                     int64_t *__restrict__ out_index, double *__restrict__ weight) {
  for (int64_t s = blockIdx.x; s < segments; s += gridDim.x) {
    const int64_t dst = seg[4 * s], q0 = seg[4 * s + 1], n = seg[4 * s + 2], p = seg[4 * s + 3];
    const bool narrow = ptrs[4 * p + 3] == 2;
    const int32_t *pin = reinterpret_cast<const int32_t *>(ptrs[4 * p]);
    const int32_t *pout = reinterpret_cast<const int32_t *>(ptrs[4 * p + 1]);
    const uint16_t *pin16 = reinterpret_cast<const uint16_t *>(ptrs[4 * p]);
    const uint16_t *pout16 = reinterpret_cast<const uint16_t *>(ptrs[4 * p + 1]);
    const double *pw = reinterpret_cast<const double *>(ptrs[4 * p + 2]);
    const int64_t *li = leaves_in + bases[2 * p];
    const int64_t *lo = leaves_out + bases[2 * p + 1];
    for (int64_t j = threadIdx.x; j < n; j += blockDim.x) {
      in_index[dst + j] = li[narrow ? int64_t(pin16[q0 + j]) : int64_t(pin[q0 + j])];
      out_index[dst + j] = lo[narrow ? int64_t(pout16[q0 + j]) : int64_t(pout[q0 + j])];
      if (weight)
        weight[dst + j] = pw ? pw[q0 + j] : 1.0;
    }
  }
}

} // namespace

void cuda_compose_parts(const at::Tensor &table, int64_t segments, int64_t parts,
                        int64_t n_in, int64_t n_out, at::Tensor &in_index, at::Tensor &out_index,
                        at::Tensor *weight) {
  TORCH_CHECK(table.is_cuda() && table.scalar_type() == at::kLong && table.is_contiguous(),
              "cuda_compose_parts: table must be a contiguous int64 CUDA tensor");
  TORCH_CHECK(table.numel() == 4 * segments + 6 * parts + n_in + n_out,
              "cuda_compose_parts: table size mismatch");
  if (segments == 0)
    return;
  const int64_t *t = table.data_ptr<int64_t>();
  const int64_t *seg = t;
  const int64_t *ptrs = seg + 4 * segments;
  const int64_t *bases = ptrs + 4 * parts;
  const int64_t *leaves_in = bases + 2 * parts;
  const int64_t *leaves_out = leaves_in + n_in;
  const int blocks = static_cast<int>(std::min<int64_t>(segments, 65535));
  auto stream = at::cuda::getCurrentCUDAStream(table.device().index());
  compose_parts_kernel<<<blocks, 128, 0, stream>>>(
      seg, segments, ptrs, bases, leaves_in, leaves_out, in_index.data_ptr<int64_t>(),
      out_index.data_ptr<int64_t>(), weight ? weight->data_ptr<double>() : nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace detail
} // namespace adaptive
} // namespace minkowski
