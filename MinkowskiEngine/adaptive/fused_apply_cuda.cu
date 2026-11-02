// Fused gather-GEMM-scatter kernels for the CUDA adaptive apply_map (see fused_apply.hpp).
// Same tile structure as the engine's per-offset `matmul` kernels (convolution_kernel.cu),
// but one launch covers every kernel slot: blockIdx.x walks a TileTable's (begin, end, slot)
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

// TileTable blocks from its slot offsets and block-count prefixes (one thread per block).
__global__ void expand_tiles(const int64_t *__restrict__ slots, int nk, int64_t count,
                             int64_t rows, int64_t *__restrict__ blocks) {
  const int64_t t = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (t >= count)
    return;
  const int64_t *off = slots, *pre = slots + nk + 1;
  int lo = 0, hi = nk - 1; // largest k with pre[k] <= t (empty slots repeat a prefix)
  while (lo < hi) {
    const int mid = (lo + hi + 1) / 2;
    if (pre[mid] <= t)
      lo = mid;
    else
      hi = mid - 1;
  }
  const int64_t begin = off[lo] + (t - pre[lo]) * rows;
  blocks[3 * t] = begin;
  blocks[3 * t + 1] = min(begin + rows, off[lo + 1]);
  blocks[3 * t + 2] = lo;
}

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

// Register-blocked variants for wider channels (fused_tile == 32): TM x TN threads compute a
// (4 TM) x (4 TN) output tile, each thread a 4 x 4 micro-tile (rows tr + TM i, columns
// tc + TN j), over K steps of 16. A side of 32 or fewer channels uses a 32-wide tile instead of
// 64, so no lanes idle. Loads are coalesced along the contiguous dimension of each operand
// (also for the transposed weights of the input gradient); shared memory is padded.
constexpr int RB_K = 16;
constexpr int RB_ROWS = 64; // pairs per forward / input-gradient tile (TM = 16)

template <typename T, int TM, int TN>
__global__ void __launch_bounds__(TM *TN)
    gather_gemm_scatter_rb(const T *__restrict__ x, const T *__restrict__ w,
                           const int64_t *__restrict__ from, const int64_t *__restrict__ to,
                           const T *__restrict__ pw, const int64_t *__restrict__ tiles, int cin,
                           int cout, bool trans_w, T *__restrict__ y) {
  constexpr int M = 4 * TM, N = 4 * TN, THREADS = TM * TN, QA = M * RB_K / THREADS,
                QB = N * RB_K / THREADS;
  static_assert(THREADS % RB_K == 0 && QA >= 1 && QB >= 1, "tile shape");
  __shared__ T As[RB_K][M + 1];
  __shared__ T Bs[RB_K][N + 1];
  const int tid = threadIdx.x, tr = tid / TN, tc = tid % TN;
  const int64_t *tile = tiles + 3 * static_cast<int64_t>(blockIdx.x);
  const int64_t begin = tile[0], end = tile[1];
  const T *wk = w + tile[2] * static_cast<int64_t>(cin) * cout;
  const int n0 = blockIdx.y * N;
  // This thread loads A rows tid / 16 + (THREADS / 16) q at column tid % 16 of every K step.
  const int a_k = tid % RB_K;
  int64_t a_src[QA];
  T a_scale[QA];
#pragma unroll
  for (int q = 0; q < QA; ++q) {
    const int64_t row = begin + tid / RB_K + (THREADS / RB_K) * q;
    const bool ok = row < end;
    a_src[q] = ok ? from[row] * cin : -1;
    a_scale[q] = ok ? (pw ? pw[row] : T(1)) : T(0);
  }
  T acc[4][4] = {};
  for (int c0 = 0; c0 < cin; c0 += RB_K) {
    const int c = c0 + a_k;
#pragma unroll
    for (int q = 0; q < QA; ++q)
      As[a_k][tid / RB_K + (THREADS / RB_K) * q] =
          (a_src[q] >= 0 && c < cin) ? x[a_src[q] + c] * a_scale[q] : T(0);
#pragma unroll
    for (int q = 0; q < QB; ++q) {
      const int e = tid + THREADS * q;
      if (!trans_w) { // W[k] is [cin, cout]: contiguous along the output columns
        const int kk = e / N, n = e % N, cc = c0 + kk, col = n0 + n;
        Bs[kk][n] = (cc < cin && col < cout) ? wk[static_cast<int64_t>(cc) * cout + col] : T(0);
      } else { // W[k] is [cout, cin] read transposed: contiguous along K
        const int n = e / RB_K, kk = e % RB_K, cc = c0 + kk, col = n0 + n;
        Bs[kk][n] = (cc < cin && col < cout) ? wk[static_cast<int64_t>(col) * cin + cc] : T(0);
      }
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < RB_K; ++kk) {
      T a[4], b[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        a[i] = As[kk][tr + TM * i];
        b[i] = Bs[kk][tc + TN * i];
      }
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
          acc[i][j] += a[i] * b[j];
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int64_t row = begin + tr + TM * i;
    if (row >= end)
      continue;
    T *out = y + to[row] * cout;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int col = n0 + tc + TN * j;
      if (col < cout)
        atomicAdd(out + col, acc[i][j]);
    }
  }
}

// gw[k] tile [c0, c0 + 4 TM) x [j0, j0 + 4 TN) over one chunk of pairs, 16 rows per step.
template <typename T, int TM, int TN>
__global__ void __launch_bounds__(TM *TN)
    weight_grad_rb(const T *__restrict__ x, const T *__restrict__ gy,
                   const int64_t *__restrict__ from, const int64_t *__restrict__ to,
                   const T *__restrict__ pw, const int64_t *__restrict__ chunks, int cin, int cout,
                   T *__restrict__ gw) {
  constexpr int M = 4 * TM, N = 4 * TN, THREADS = TM * TN;
  __shared__ T Xs[RB_K][M + 1];
  __shared__ T Gs[RB_K][N + 1];
  const int tid = threadIdx.x, tr = tid / TN, tc = tid % TN;
  const int64_t *chunk = chunks + 3 * static_cast<int64_t>(blockIdx.x);
  const int64_t begin = chunk[0], end = chunk[1];
  const int c0 = blockIdx.y * M, j0 = blockIdx.z * N;
  T acc[4][4] = {};
  for (int64_t r0 = begin; r0 < end; r0 += RB_K) {
    for (int e = tid; e < RB_K * M; e += THREADS) {
      const int kk = e / M, n = e % M;
      const int64_t row = r0 + kk;
      Xs[kk][n] = (row < end && c0 + n < cin)
                      ? x[from[row] * cin + c0 + n] * (pw ? pw[row] : T(1))
                      : T(0);
    }
    for (int e = tid; e < RB_K * N; e += THREADS) {
      const int kk = e / N, n = e % N;
      const int64_t row = r0 + kk;
      Gs[kk][n] = (row < end && j0 + n < cout) ? gy[to[row] * cout + j0 + n] : T(0);
    }
    __syncthreads();
    // Per-step partial sums (see weight_grad).
    T part[4][4] = {};
#pragma unroll
    for (int kk = 0; kk < RB_K; ++kk) {
      T a[4], b[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        a[i] = Xs[kk][tr + TM * i];
        b[i] = Gs[kk][tc + TN * i];
      }
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
          part[i][j] += a[i] * b[j];
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j)
        acc[i][j] += part[i][j];
    __syncthreads();
  }
  T *g = gw + chunk[2] * static_cast<int64_t>(cin) * cout;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int c = c0 + tr + TM * i;
    if (c >= cin)
      continue;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int jj = j0 + tc + TN * j;
      if (jj < cout)
        atomicAdd(g + static_cast<int64_t>(c) * cout + jj, acc[i][j]);
    }
  }
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
                    const at::Tensor &to, const at::Tensor &pw, const TileTable &tiles,
                    bool trans_w, at::Tensor &y, int cin, int cout) {
  const dim3 block(BS, BS);
  const dim3 grid(static_cast<unsigned>(tiles.count), static_cast<unsigned>((cout + BS - 1) / BS));
  gather_gemm_scatter<T, BS><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<T>(), w.data_ptr<T>(), from.data_ptr<int64_t>(), to.data_ptr<int64_t>(),
      maybe<T>(pw), tiles.blocks.data_ptr<int64_t>(), cin, cout, trans_w, y.data_ptr<T>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T, int BS>
void launch_weight_grad(const at::Tensor &x, const at::Tensor &gy, const at::Tensor &from,
                        const at::Tensor &to, const at::Tensor &pw, const TileTable &chunks,
                        at::Tensor &gw, int cin, int cout) {
  const dim3 block(BS, BS);
  const dim3 grid(static_cast<unsigned>(chunks.count), static_cast<unsigned>((cin + BS - 1) / BS),
                  static_cast<unsigned>((cout + BS - 1) / BS));
  weight_grad<T, BS><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<T>(), gy.data_ptr<T>(), from.data_ptr<int64_t>(), to.data_ptr<int64_t>(),
      maybe<T>(pw), chunks.blocks.data_ptr<int64_t>(), cin, cout, gw.data_ptr<T>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// 16 thread rows/columns for a 64-wide side, 8 for a side of at most 32 channels.
inline int rb_threads(int channels) { return channels <= 32 ? 8 : 16; }

template <typename T, int TM, int TN>
void launch_forward_rb_t(const at::Tensor &x, const at::Tensor &w, const at::Tensor &from,
                         const at::Tensor &to, const at::Tensor &pw, const TileTable &tiles,
                         bool trans_w, at::Tensor &y, int cin, int cout) {
  const dim3 grid(static_cast<unsigned>(tiles.count),
                  static_cast<unsigned>((cout + 4 * TN - 1) / (4 * TN)));
  gather_gemm_scatter_rb<T, TM, TN><<<grid, TM * TN, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<T>(), w.data_ptr<T>(), from.data_ptr<int64_t>(), to.data_ptr<int64_t>(),
      maybe<T>(pw), tiles.blocks.data_ptr<int64_t>(), cin, cout, trans_w, y.data_ptr<T>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void launch_forward_rb(const at::Tensor &x, const at::Tensor &w, const at::Tensor &from,
                       const at::Tensor &to, const at::Tensor &pw, const TileTable &tiles,
                       bool trans_w, at::Tensor &y, int cin, int cout) {
  // Rows always 64 (RB_ROWS, the tile table's row count); the output side may be 32 wide.
  if (rb_threads(cout) == 8)
    launch_forward_rb_t<T, 16, 8>(x, w, from, to, pw, tiles, trans_w, y, cin, cout);
  else
    launch_forward_rb_t<T, 16, 16>(x, w, from, to, pw, tiles, trans_w, y, cin, cout);
}

template <typename T, int TM, int TN>
void launch_weight_grad_rb_t(const at::Tensor &x, const at::Tensor &gy, const at::Tensor &from,
                             const at::Tensor &to, const at::Tensor &pw,
                             const TileTable &chunks, at::Tensor &gw, int cin, int cout) {
  const dim3 grid(static_cast<unsigned>(chunks.count),
                  static_cast<unsigned>((cin + 4 * TM - 1) / (4 * TM)),
                  static_cast<unsigned>((cout + 4 * TN - 1) / (4 * TN)));
  weight_grad_rb<T, TM, TN><<<grid, TM * TN, 0, at::cuda::getCurrentCUDAStream()>>>(
      x.data_ptr<T>(), gy.data_ptr<T>(), from.data_ptr<int64_t>(), to.data_ptr<int64_t>(),
      maybe<T>(pw), chunks.blocks.data_ptr<int64_t>(), cin, cout, gw.data_ptr<T>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void launch_weight_grad_rb(const at::Tensor &x, const at::Tensor &gy, const at::Tensor &from,
                           const at::Tensor &to, const at::Tensor &pw, const TileTable &chunks,
                           at::Tensor &gw, int cin, int cout) {
  const int tm = rb_threads(cin), tn = rb_threads(cout);
  if (tm == 8 && tn == 8)
    launch_weight_grad_rb_t<T, 8, 8>(x, gy, from, to, pw, chunks, gw, cin, cout);
  else if (tm == 8)
    launch_weight_grad_rb_t<T, 8, 16>(x, gy, from, to, pw, chunks, gw, cin, cout);
  else if (tn == 8)
    launch_weight_grad_rb_t<T, 16, 8>(x, gy, from, to, pw, chunks, gw, cin, cout);
  else
    launch_weight_grad_rb_t<T, 16, 16>(x, gy, from, to, pw, chunks, gw, cin, cout);
}

template <typename F>
void with_tile(int bs, F &&f) {
  if (bs == 8)
    f(std::integral_constant<int, 8>());
  else
    f(std::integral_constant<int, 16>());
}

} // namespace

at::Tensor cuda_expand_tiles(const at::Tensor &slots, int64_t num_kernels, int64_t count,
                             int64_t rows) {
  auto blocks = at::empty({count, 3}, slots.options());
  if (count == 0)
    return blocks;
  const c10::cuda::CUDAGuard guard(slots.device());
  constexpr int threads = 256;
  expand_tiles<<<static_cast<unsigned>((count + threads - 1) / threads), threads, 0,
                 at::cuda::getCurrentCUDAStream()>>>(slots.data_ptr<int64_t>(),
                                                     static_cast<int>(num_kernels), count, rows,
                                                     blocks.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return blocks;
}

int fused_tile_rows(int64_t cin, int64_t cout) {
  const int bs = fused_tile(cin, cout);
  return bs == 32 ? RB_ROWS : bs;
}

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
                                    const at::Tensor &pw, const TileTable &tiles, bool trans_w,
                                    at::Tensor &y) {
  check_args(x, w, from, to, pw, tiles.blocks, y);
  if (tiles.count == 0 || y.size(1) == 0)
    return;
  if (tiles.count > 0x7fffffff)
    throw std::invalid_argument("fused apply_map: too many tiles for one launch");
  const c10::cuda::CUDAGuard guard(x.device());
  const int cin = static_cast<int>(x.size(1)), cout = static_cast<int>(y.size(1));
  const int bs = fused_tile(cin, cout);
  if (bs == 32) {
    if (x.scalar_type() == at::kDouble)
      launch_forward_rb<double>(x, w, from, to, pw, tiles, trans_w, y, cin, cout);
    else if (x.scalar_type() == at::kFloat)
      launch_forward_rb<float>(x, w, from, to, pw, tiles, trans_w, y, cin, cout);
    else
      throw std::invalid_argument("fused apply_map: float or double only");
    return;
  }
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
                            const at::Tensor &to, const at::Tensor &pw, const TileTable &chunks,
                            at::Tensor &gw) {
  check_args(x, gy, from, to, pw, chunks.blocks, gw);
  if (chunks.count == 0 || x.size(1) == 0 || gy.size(1) == 0)
    return;
  if (chunks.count > 0x7fffffff)
    throw std::invalid_argument("fused apply_map: too many chunks for one launch");
  const c10::cuda::CUDAGuard guard(x.device());
  const int cin = static_cast<int>(x.size(1)), cout = static_cast<int>(gy.size(1));
  const int bs = fused_tile(cin, cout);
  if (bs == 32) {
    if (x.scalar_type() == at::kDouble)
      launch_weight_grad_rb<double>(x, gy, from, to, pw, chunks, gw, cin, cout);
    else if (x.scalar_type() == at::kFloat)
      launch_weight_grad_rb<float>(x, gy, from, to, pw, chunks, gw, cin, cout);
    else
      throw std::invalid_argument("fused apply_map: float or double only");
    return;
  }
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
