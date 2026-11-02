#ifndef MINK_CPP_ADAPTIVE_FUSED_APPLY_HPP
#define MINK_CPP_ADAPTIVE_FUSED_APPLY_HPP

// Single-launch CUDA kernels for applying an adaptive kernel map (fused_apply_cuda.cu).
// The map's pairs are sorted by kernel slot; a "tile table" ([T, 3] int64 on the device:
// begin, end, slot) cuts every slot's pair range into row blocks, so one launch covers all
// slots with no gathered or product temporaries (KernelMapDeviceCache::tiles).
#ifdef USE_CUDA

#include <torch/torch.h>

#include <cstdint>

namespace minkowski {
namespace adaptive {
namespace detail {

// Tile edge for these channel counts (8, 16 or 32; symmetric in cin/cout). 8 and 16 use
// BS x BS thread blocks; 32 selects the register-blocked 64 x 64 kernels.
int fused_tile(int64_t cin, int64_t cout);
// Pairs per tile the forward/input-gradient tile table must use (BS, or 64 when blocked).
int fused_tile_rows(int64_t cin, int64_t cout);

// y[to[p]] += pw[p] * x[from[p]] @ W[k(p)] over all pairs p (pw empty = weight 1).
// w is [K, cin, cout], or [K, cout, cin] read transposed when `trans_w` (input gradient).
// x [*, cin], y [*, cout] contiguous, same dtype (float or double) and device. `tiles` must
// use fused_tile_rows(cin, cout) rows per tile.
void cuda_fused_gather_gemm_scatter(const at::Tensor &x, const at::Tensor &w,
                                    const at::Tensor &from, const at::Tensor &to,
                                    const at::Tensor &pw, const at::Tensor &tiles, bool trans_w,
                                    at::Tensor &y);

// gw[k] += sum over the pairs p of slot k of pw[p] * x[from[p]]^T gy[to[p]] (gw [K, cin, cout]).
// `chunks` is a tile table with any number of rows per tile (each tile is one atomic partial).
void cuda_fused_weight_grad(const at::Tensor &x, const at::Tensor &gy, const at::Tensor &from,
                            const at::Tensor &to, const at::Tensor &pw, const at::Tensor &chunks,
                            at::Tensor &gw);

} // namespace detail
} // namespace adaptive
} // namespace minkowski

#endif // USE_CUDA
#endif
