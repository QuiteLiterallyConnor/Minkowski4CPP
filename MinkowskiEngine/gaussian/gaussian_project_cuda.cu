// Fused CUDA projection of 3D Gaussians to screen space (render_gaussians, see gaussian_render.cpp).
// One thread per Gaussian computes what the LibTorch path computes with ~60 element-wise ops: the
// camera-space point, the clamped-frustum Jacobian, the 2D covariance J W R(q) S (J W R(q) S)^T with
// the low-pass term, its determinant (Lagrange identity) and the conic. The backward kernel
// recomputes these values and applies the chain rule analytically, so the autograd graph of the
// projection is one node instead of ~100.
#include "gaussian_cuda.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

namespace minkowski {
namespace gaussian {
namespace detail {
namespace {

constexpr float kLowPassF = 0.3f;
constexpr float kLowPass2F = 0.09f;
constexpr float kMinDetF = 1e-12f;
constexpr float kMinNormF = 1e-12f;

struct Forward {
  // inputs
  float p[3], q[4], s[3];
  // intermediates
  float cam[3], sz, tx, ty, txc, tyc;
  float n[4], qnorm;
  float Rq[3][3], B[3][3], T[2][3], m1[3], m2[3], cross[3];
  float cov_a, b, cov_c, a, c, det, dclamp, inv;
  float x2, y2;
};

// W (rows 0-2 of world_to_camera, [3,4] row-major: rotation | translation).
__device__ __forceinline__ void project_one(Forward &f, const float *__restrict__ W,
                                            const ProjectParams &k) {
  for (int i = 0; i < 3; ++i)
    f.cam[i] = W[4 * i] * f.p[0] + W[4 * i + 1] * f.p[1] + W[4 * i + 2] * f.p[2] + W[4 * i + 3];
  const float z = f.cam[2];
  f.sz = z < k.near_plane ? k.near_plane : z; // clamp_min; NaN passes through
  f.tx = f.cam[0] / f.sz;
  f.ty = f.cam[1] / f.sz;
  f.x2 = k.fx * f.tx + k.cx;
  f.y2 = k.fy * f.ty + k.cy;
  f.txc = f.tx < k.tx_lo ? k.tx_lo : (f.tx > k.tx_hi ? k.tx_hi : f.tx);
  f.tyc = f.ty < k.ty_lo ? k.ty_lo : (f.ty > k.ty_hi ? k.ty_hi : f.ty);
  const float J00 = k.fx / f.sz, J02 = (-k.fx * f.txc) / f.sz;
  const float J11 = k.fy / f.sz, J12 = (-k.fy * f.tyc) / f.sz;
  for (int j = 0; j < 3; ++j) {
    f.T[0][j] = J00 * W[j] + J02 * W[8 + j];
    f.T[1][j] = J11 * W[4 + j] + J12 * W[8 + j];
  }
  f.qnorm = sqrtf(f.q[0] * f.q[0] + f.q[1] * f.q[1] + f.q[2] * f.q[2] + f.q[3] * f.q[3]);
  const float denom = f.qnorm < kMinNormF ? kMinNormF : f.qnorm;
  for (int i = 0; i < 4; ++i)
    f.n[i] = f.q[i] / denom;
  const float w = f.n[0], x = f.n[1], y = f.n[2], zq = f.n[3];
  f.Rq[0][0] = 1 - 2 * (y * y + zq * zq);
  f.Rq[0][1] = 2 * (x * y - w * zq);
  f.Rq[0][2] = 2 * (x * zq + w * y);
  f.Rq[1][0] = 2 * (x * y + w * zq);
  f.Rq[1][1] = 1 - 2 * (x * x + zq * zq);
  f.Rq[1][2] = 2 * (y * zq - w * x);
  f.Rq[2][0] = 2 * (x * zq - w * y);
  f.Rq[2][1] = 2 * (y * zq + w * x);
  f.Rq[2][2] = 1 - 2 * (x * x + y * y);
  for (int j = 0; j < 3; ++j)
    for (int kk = 0; kk < 3; ++kk)
      f.B[j][kk] = f.Rq[j][kk] * f.s[kk];
  for (int kk = 0; kk < 3; ++kk) {
    f.m1[kk] = f.T[0][0] * f.B[0][kk] + f.T[0][1] * f.B[1][kk] + f.T[0][2] * f.B[2][kk];
    f.m2[kk] = f.T[1][0] * f.B[0][kk] + f.T[1][1] * f.B[1][kk] + f.T[1][2] * f.B[2][kk];
  }
  f.cov_a = f.m1[0] * f.m1[0] + f.m1[1] * f.m1[1] + f.m1[2] * f.m1[2];
  f.b = f.m1[0] * f.m2[0] + f.m1[1] * f.m2[1] + f.m1[2] * f.m2[2];
  f.cov_c = f.m2[0] * f.m2[0] + f.m2[1] * f.m2[1] + f.m2[2] * f.m2[2];
  f.a = f.cov_a + kLowPassF;
  f.c = f.cov_c + kLowPassF;
  f.cross[0] = f.m1[1] * f.m2[2] - f.m1[2] * f.m2[1];
  f.cross[1] = f.m1[2] * f.m2[0] - f.m1[0] * f.m2[2];
  f.cross[2] = f.m1[0] * f.m2[1] - f.m1[1] * f.m2[0];
  f.det = (f.cross[0] * f.cross[0] + f.cross[1] * f.cross[1] + f.cross[2] * f.cross[2]) +
          kLowPassF * (f.cov_a + f.cov_c) + kLowPass2F;
  f.dclamp = f.det < kMinDetF ? kMinDetF : f.det;
  f.inv = 1.0f / f.dclamp;
}

__device__ __forceinline__ void load(Forward &f, const float *P, const float *Q, const float *S,
                                     int64_t i) {
  for (int d = 0; d < 3; ++d) {
    f.p[d] = P[3 * i + d];
    f.s[d] = S[3 * i + d];
  }
  for (int d = 0; d < 4; ++d)
    f.q[d] = Q[4 * i + d];
}

__global__ void project_forward_kernel(int64_t n, const float *__restrict__ P,
                                       const float *__restrict__ Q, const float *__restrict__ S,
                                       const float *__restrict__ W, ProjectParams k,
                                       float *__restrict__ out) {
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= n)
    return;
  Forward f;
  load(f, P, Q, S, i);
  project_one(f, W, k);
  float *o = out + kProjectColumns * i;
  o[0] = f.x2;
  o[1] = f.y2;
  o[2] = f.c * f.inv;
  o[3] = -f.b * f.inv;
  o[4] = f.a * f.inv;
  o[5] = f.cam[2];
  o[6] = f.a;
  o[7] = f.b;
  o[8] = f.c;
  o[9] = f.det;
}

__global__ void project_backward_kernel(int64_t n, const float *__restrict__ P,
                                        const float *__restrict__ Q, const float *__restrict__ S,
                                        const float *__restrict__ W, ProjectParams k,
                                        const float *__restrict__ grad,
                                        float *__restrict__ gP, float *__restrict__ gQ,
                                        float *__restrict__ gS) {
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i >= n)
    return;
  Forward f;
  load(f, P, Q, S, i);
  project_one(f, W, k);
  const float *g = grad + kProjectColumns * i;
  const float g_x2 = g[0], g_y2 = g[1], g_ca = g[2], g_cb = g[3], g_cc = g[4], g_z = g[5];

  // conic = (c, -b, a) / max(det, eps)
  const float d_c = g_ca * f.inv, d_b = -g_cb * f.inv, d_a = g_cc * f.inv;
  const float d_clamp = -(g_ca * f.c - g_cb * f.b + g_cc * f.a) * f.inv * f.inv;
  const float d_det = f.det >= kMinDetF ? d_clamp : 0.0f;
  const float d_cov_a = d_a + kLowPassF * d_det, d_cov_c = d_c + kLowPassF * d_det;
  float dcross[3];
  for (int d = 0; d < 3; ++d)
    dcross[d] = 2.0f * f.cross[d] * d_det;
  // u = m1 x m2: du/dm1 . g = m2 x g, du/dm2 . g = g x m1
  float dm1[3], dm2[3];
  dm1[0] = f.m2[1] * dcross[2] - f.m2[2] * dcross[1];
  dm1[1] = f.m2[2] * dcross[0] - f.m2[0] * dcross[2];
  dm1[2] = f.m2[0] * dcross[1] - f.m2[1] * dcross[0];
  dm2[0] = dcross[1] * f.m1[2] - dcross[2] * f.m1[1];
  dm2[1] = dcross[2] * f.m1[0] - dcross[0] * f.m1[2];
  dm2[2] = dcross[0] * f.m1[1] - dcross[1] * f.m1[0];
  for (int d = 0; d < 3; ++d) {
    dm1[d] += 2.0f * f.m1[d] * d_cov_a + f.m2[d] * d_b;
    dm2[d] += 2.0f * f.m2[d] * d_cov_c + f.m1[d] * d_b;
  }
  // [m1; m2] = T B
  float dT[2][3], dB[3][3];
  for (int j = 0; j < 3; ++j) {
    dT[0][j] = dm1[0] * f.B[j][0] + dm1[1] * f.B[j][1] + dm1[2] * f.B[j][2];
    dT[1][j] = dm2[0] * f.B[j][0] + dm2[1] * f.B[j][1] + dm2[2] * f.B[j][2];
    for (int kk = 0; kk < 3; ++kk)
      dB[j][kk] = f.T[0][j] * dm1[kk] + f.T[1][j] * dm2[kk];
  }
  // B = Rq diag(s)
  float ds[3] = {0, 0, 0}, dR[3][3];
  for (int j = 0; j < 3; ++j)
    for (int kk = 0; kk < 3; ++kk) {
      ds[kk] += dB[j][kk] * f.Rq[j][kk];
      dR[j][kk] = dB[j][kk] * f.s[kk];
    }
  const float w = f.n[0], x = f.n[1], y = f.n[2], zq = f.n[3];
  float dn[4];
  dn[0] = 2 * (-zq * dR[0][1] + y * dR[0][2] + zq * dR[1][0] - x * dR[1][2] - y * dR[2][0] +
               x * dR[2][1]);
  dn[1] = 2 * (y * dR[0][1] + zq * dR[0][2] + y * dR[1][0] - 2 * x * dR[1][1] - w * dR[1][2] +
               zq * dR[2][0] + w * dR[2][1] - 2 * x * dR[2][2]);
  dn[2] = 2 * (-2 * y * dR[0][0] + x * dR[0][1] + w * dR[0][2] + x * dR[1][0] + zq * dR[1][2] -
               w * dR[2][0] + zq * dR[2][1] - 2 * y * dR[2][2]);
  dn[3] = 2 * (-2 * zq * dR[0][0] - w * dR[0][1] + x * dR[0][2] + w * dR[1][0] -
               2 * zq * dR[1][1] + y * dR[1][2] + x * dR[2][0] + y * dR[2][1]);
  // n = q / max(|q|, eps)
  float dq[4];
  if (f.qnorm >= kMinNormF) {
    const float nd = f.n[0] * dn[0] + f.n[1] * dn[1] + f.n[2] * dn[2] + f.n[3] * dn[3];
    for (int d = 0; d < 4; ++d)
      dq[d] = (dn[d] - f.n[d] * nd) / f.qnorm;
  } else {
    for (int d = 0; d < 4; ++d)
      dq[d] = dn[d] / kMinNormF;
  }
  // T = J W (J has zeros at [0][1] and [1][0])
  float dJ00 = 0, dJ02 = 0, dJ11 = 0, dJ12 = 0;
  for (int kk = 0; kk < 3; ++kk) {
    dJ00 += dT[0][kk] * W[kk];
    dJ02 += dT[0][kk] * W[8 + kk];
    dJ11 += dT[1][kk] * W[4 + kk];
    dJ12 += dT[1][kk] * W[8 + kk];
  }
  const float sz2 = f.sz * f.sz;
  float d_sz = dJ00 * (-k.fx / sz2) + dJ02 * (k.fx * f.txc / sz2) + dJ11 * (-k.fy / sz2) +
               dJ12 * (k.fy * f.tyc / sz2);
  const float d_txc = dJ02 * (-k.fx / f.sz), d_tyc = dJ12 * (-k.fy / f.sz);
  float d_tx = k.fx * g_x2 + ((f.tx >= k.tx_lo && f.tx <= k.tx_hi) ? d_txc : 0.0f);
  float d_ty = k.fy * g_y2 + ((f.ty >= k.ty_lo && f.ty <= k.ty_hi) ? d_tyc : 0.0f);
  float dcam[3];
  dcam[0] = d_tx / f.sz;
  dcam[1] = d_ty / f.sz;
  d_sz += -d_tx * f.cam[0] / sz2 - d_ty * f.cam[1] / sz2;
  dcam[2] = g_z + (f.cam[2] >= k.near_plane ? d_sz : 0.0f);
  float *gp = gP + 3 * i, *gq = gQ + 4 * i, *gs = gS + 3 * i;
  for (int j = 0; j < 3; ++j)
    gp[j] = W[j] * dcam[0] + W[4 + j] * dcam[1] + W[8 + j] * dcam[2];
  for (int d = 0; d < 4; ++d)
    gq[d] = dq[d];
  for (int d = 0; d < 3; ++d)
    gs[d] = ds[d];
}

void check(const at::Tensor &t, int64_t n, int64_t cols, const char *name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kFloat && t.is_contiguous() &&
                  t.dim() == 2 && t.size(0) == n && t.size(1) == cols,
              "project_cuda: ", name, " must be a contiguous float32 CUDA tensor [N, ", cols, "]");
}

} // namespace

void project_forward_cuda(const at::Tensor &positions, const at::Tensor &rotations,
                          const at::Tensor &scales, const at::Tensor &world_to_camera,
                          const ProjectParams &params, at::Tensor &out) {
  const int64_t n = positions.size(0);
  check(positions, n, 3, "positions");
  check(rotations, n, 4, "rotations");
  check(scales, n, 3, "scales");
  check(out, n, kProjectColumns, "out");
  check(world_to_camera, 3, 4, "world_to_camera");
  if (n == 0)
    return;
  const c10::cuda::CUDAGuard guard(positions.device());
  constexpr int threads = 128;
  project_forward_kernel<<<static_cast<unsigned>((n + threads - 1) / threads), threads, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
      n, positions.data_ptr<float>(), rotations.data_ptr<float>(), scales.data_ptr<float>(),
      world_to_camera.data_ptr<float>(), params, out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void project_backward_cuda(const at::Tensor &positions, const at::Tensor &rotations,
                           const at::Tensor &scales, const at::Tensor &world_to_camera,
                           const ProjectParams &params, const at::Tensor &grad_out,
                           at::Tensor &grad_positions, at::Tensor &grad_rotations,
                           at::Tensor &grad_scales) {
  const int64_t n = positions.size(0);
  check(positions, n, 3, "positions");
  check(rotations, n, 4, "rotations");
  check(scales, n, 3, "scales");
  check(world_to_camera, 3, 4, "world_to_camera");
  check(grad_out, n, kProjectColumns, "grad_out");
  check(grad_positions, n, 3, "grad_positions");
  check(grad_rotations, n, 4, "grad_rotations");
  check(grad_scales, n, 3, "grad_scales");
  if (n == 0)
    return;
  const c10::cuda::CUDAGuard guard(positions.device());
  constexpr int threads = 128;
  project_backward_kernel<<<static_cast<unsigned>((n + threads - 1) / threads), threads, 0,
                            at::cuda::getCurrentCUDAStream()>>>(
      n, positions.data_ptr<float>(), rotations.data_ptr<float>(), scales.data_ptr<float>(),
      world_to_camera.data_ptr<float>(), params, grad_out.data_ptr<float>(),
      grad_positions.data_ptr<float>(), grad_rotations.data_ptr<float>(),
      grad_scales.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace detail
} // namespace gaussian
} // namespace minkowski
