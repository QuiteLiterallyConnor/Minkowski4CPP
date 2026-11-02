// Fused gather-GEMM-scatter kernels for the CUDA adaptive apply_map (see fused_apply.hpp).
// Same tile structure as the engine's per-offset `matmul` kernels (convolution_kernel.cu),
// but one launch covers every kernel slot: blockIdx.x walks a tile table of (begin, end, slot)
// row blocks, indices are int64, rows may carry a per-pair weight, and results are
// atomically added (so the summation order, not the math, is nondeterministic).
#include "fused_apply.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <stdexcept>

namespace minkowski {
namespace adaptive {
namespace detail {
namespace {

// One block = BS pairs (rows) x BS output channels. A row of x is scaled by its pair weight
// on load, multiplied with W[k] in BS-wide chunks of the input channels, and the BS x BS
// result is atomically added to the output rows.
template <typename T, int BS>
__global__ void gather_gemm_scatter(const T *__restrict__ x, const T *__restrict__ w,
                                    const int64_t *__restrict__ from,
                                    const int64_t *__restrict__ to, const T *__restrict__ pw,
                                    const int64_t *__restrict__ tiles, int cin, int cout,
                                    bool trans_w, T *__restrict__ y) {
  __shared__ T As[BS][BS + 1];
  __shared__ T Bs[BS][BS + 1];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int64_t *tile = tiles + 3 * static_cast<int64_t>(blockIdx.x);
  const int64_t row = tile[0] + ty;
  const bool row_ok = row < tile[1];
  const T *wk = w + tile[2] * static_cast<int64_t>(cin) * cout;
  const int col = blockIdx.y * BS + tx;
  const int64_t src = row_ok ? from[row] : 0;
  const T scale = row_ok ? (pw ? pw[row] : T(1)) : T(0);
  T acc = 0;
  for (int c0 = 0; c0 < cin; c0 += BS) {
    const int ca = c0 + tx, cb = c0 + ty;
    As[ty][tx] = (row_ok && ca < cin) ? x[src * cin + ca] * scale : T(0);
    Bs[ty][tx] = (cb < cin && col < cout)
                     ? (trans_w ? wk[static_cast<int64_t>(col) * cin + cb]
                                : wk[static_cast<int64_t>(cb) * cout + col])
                     : T(0);
    __syncthreads();
#pragma unroll
    for (int i = 0; i < BS; ++i)
      acc += As[ty][i] * Bs[i][tx];
    __syncthreads();
  }
  if (row_ok && col < cout)
    atomicAdd(y + to[row] * cout + col, acc);
}

// One block = one chunk of pairs of slot k x a BS x BS tile of gw[k] (input x output
// channels). The chunk is walked in BS-row steps; the partial sum is added once per element.
template <typename T, int BS>
__global__ void weight_grad(const T *__restrict__ x, const T *__restrict__ gy,
                            const int64_t *__restrict__ from, const int64_t *__restrict__ to,
                            const T *__restrict__ pw, const int64_t *__restrict__ chunks, int cin,
                            int cout, T *__restrict__ gw) {
  __shared__ T Xs[BS][BS + 1];
  __shared__ T Gs[BS][BS + 1];
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int64_t *chunk = chunks + 3 * static_cast<int64_t>(blockIdx.x);
  const int64_t begin = chunk[0], end = chunk[1];
  const int c0 = blockIdx.y * BS, j0 = blockIdx.z * BS;
  T acc = 0;
  for (int64_t r0 = begin; r0 < end; r0 += BS) {
    const int64_t row = r0 + ty;
    const bool row_ok = row < end;
    const T scale = row_ok ? (pw ? pw[row] : T(1)) : T(0);
    Xs[ty][tx] = (row_ok && c0 + tx < cin) ? x[from[row] * cin + c0 + tx] * scale : T(0);
    Gs[ty][tx] = (row_ok && j0 + tx < cout) ? gy[to[row] * cout + j0 + tx] : T(0);
    __syncthreads();
    // Sum each BS-row step on its own before adding it: the float rounding then grows with
    // BS + steps instead of the chunk length (3x closer to cuBLAS's per-slot mm in float32).
    T part = 0;
#pragma unroll
    for (int i = 0; i < BS; ++i)
      part += Xs[i][ty] * Gs[i][tx];
    acc += part;
    __syncthreads();
  }
  const int c = c0 + ty, j = j0 + tx;
  if (c < cin && j < cout)
    atomicAdd(gw + (chunk[2] * cin + c) * static_cast<int64_t>(cout) + j, acc);
}

void check_args(const at::Tensor &a, const at::Tensor &b, const at::Tensor &from,
                const at::Tensor &to, const at::Tensor &pw, const at::Tensor &tiles,
                const at::Tensor &out) {
  for (const at::Tensor *t : {&a, &b, &out})
    if (!t->is_cuda() || !t->is_contiguous() || t->scalar_type() != a.scalar_type())
      throw std::invalid_argument("fused apply_map: features/weights must be contiguous CUDA "
                                  "tensors of one dtype");
  for (const at::Tensor *t : {&from, &to, &tiles})
    if (!t->is_cuda() || !t->is_contiguous() || t->scalar_type() != at::kLong)
      throw std::invalid_argument("fused apply_map: index tensors must be contiguous CUDA int64");
  if (pw.numel() > 0 && (pw.numel() != from.numel() || pw.scalar_type() != a.scalar_type() ||
                         !pw.is_contiguous()))
    throw std::invalid_argument("fused apply_map: pair weights must match the pairs and dtype");
}

template <typename T>
const T *maybe(const at::Tensor &t) {
  return t.numel() > 0 ? t.data_ptr<T>() : nullptr;
}

template <typename T, int BS>
void launch_forward(const at::Tensor &x, const at::Tensor &w, const at::Tensor &from,
                    const at::Tensor &to, const at::Tensor &pw, const at::Tensor &tiles,
                    bool trans_w, at::Tensor &y, int cin, int cout) {
  const dim3 block(BS, BS);
  const dim3 grid(static_cast<unsigned>(tiles.size(0)), static_cast<unsigned>((cout + BS - 1) / BS));
  gather_gemm_scatter<T, BS><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<T>(), w.data_ptr<T>(), from.data_ptr<int64_t>(), to.data_ptr<int64_t>(),
      maybe<T>(pw), tiles.data_ptr<int64_t>(), cin, cout, trans_w, y.data_ptr<T>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T, int BS>
void launch_weight_grad(const at::Tensor &x, const at::Tensor &gy, const at::Tensor &from,
                        const at::Tensor &to, const at::Tensor &pw, const at::Tensor &chunks,
                        at::Tensor &gw, int cin, int cout) {
  const dim3 block(BS, BS);
  const dim3 grid(static_cast<unsigned>(chunks.size(0)), static_cast<unsigned>((cin + BS - 1) / BS),
                  static_cast<unsigned>((cout + BS - 1) / BS));
  weight_grad<T, BS><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<T>(), gy.data_ptr<T>(), from.data_ptr<int64_t>(), to.data_ptr<int64_t>(),
      maybe<T>(pw), chunks.data_ptr<int64_t>(), cin, cout, gw.data_ptr<T>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename F>
void with_tile(int bs, F &&f) {
  switch (bs) {
  case 8: f(std::integral_constant<int, 8>()); break;
  case 16: f(std::integral_constant<int, 16>()); break;
  default: f(std::integral_constant<int, 32>()); break;
  }
}

} // namespace

int fused_tile(int64_t cin, int64_t cout) {
  // The engine's shared-memory size rule (ConvolutionForwardKernelGPU) without its 24 case
  // (unreachable once symmetric), symmetric in the channel counts so the input-gradient
  // launch (cin and cout swapped) uses the same tiles.
  const int64_t lo = std::min(cin, cout);
  if ((lo > 16 && cin * cout >= 512) || lo > 24)
    return 32;
  if (lo > 8)
    return 16;
  return 8;
}

void cuda_fused_gather_gemm_scatter(const at::Tensor &x, const at::Tensor &w,
                                    const at::Tensor &from, const at::Tensor &to,
                                    const at::Tensor &pw, const at::Tensor &tiles, bool trans_w,
                                    at::Tensor &y) {
  check_args(x, w, from, to, pw, tiles, y);
  if (tiles.size(0) == 0 || y.size(1) == 0)
    return;
  if (tiles.size(0) > 0x7fffffff)
    throw std::invalid_argument("fused apply_map: too many tiles for one launch");
  const c10::cuda::CUDAGuard guard(x.device());
  const int cin = static_cast<int>(x.size(1)), cout = static_cast<int>(y.size(1));
  const int bs = fused_tile(cin, cout);
  with_tile(bs, [&](auto b) {
    constexpr int BS = decltype(b)::value;
    if (x.scalar_type() == at::kDouble)
      launch_forward<double, BS>(x, w, from, to, pw, tiles, trans_w, y, cin, cout);
    else if (x.scalar_type() == at::kFloat)
      launch_forward<float, BS>(x, w, from, to, pw, tiles, trans_w, y, cin, cout);
    else
      throw std::invalid_argument("fused apply_map: float or double only");
  });
}

void cuda_fused_weight_grad(const at::Tensor &x, const at::Tensor &gy, const at::Tensor &from,
                            const at::Tensor &to, const at::Tensor &pw, const at::Tensor &chunks,
                            at::Tensor &gw) {
  check_args(x, gy, from, to, pw, chunks, gw);
  if (chunks.size(0) == 0 || x.size(1) == 0 || gy.size(1) == 0)
    return;
  if (chunks.size(0) > 0x7fffffff)
    throw std::invalid_argument("fused apply_map: too many chunks for one launch");
  const c10::cuda::CUDAGuard guard(x.device());
  const int cin = static_cast<int>(x.size(1)), cout = static_cast<int>(gy.size(1));
  const int bs = fused_tile(cin, cout);
  with_tile(bs, [&](auto b) {
    constexpr int BS = decltype(b)::value;
    if (x.scalar_type() == at::kDouble)
      launch_weight_grad<double, BS>(x, gy, from, to, pw, chunks, gw, cin, cout);
    else if (x.scalar_type() == at::kFloat)
      launch_weight_grad<float, BS>(x, gy, from, to, pw, chunks, gw, cin, cout);
    else
      throw std::invalid_argument("fused apply_map: float or double only");
  });
}

} // namespace detail
} // namespace adaptive
} // namespace minkowski
