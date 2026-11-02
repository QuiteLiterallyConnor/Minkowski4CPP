/*  Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
 *
 *  Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *  The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 *  Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
 *  Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
 *  of the code.
 */
#include "math_functions.hpp"

#include <array>

#include <torch/torch.h>

// Implement the CBLAS GEMM entry points with LibTorch when MKL is not available.
// Full BLAS semantics: row- or column-major storage, transposes and leading
// dimensions; with beta == 0 the input contents of C are ignored.
#ifndef USE_MKL
namespace {

// Strided view of a stored BLAS matrix with `rows` x `cols` elements.
template <typename T>
at::Tensor blas_matrix(const T *data, int rows, int cols, int ld,
                       bool row_major) {
  const std::array<int64_t, 2> sizes{rows, cols};
  const std::array<int64_t, 2> strides =
      row_major ? std::array<int64_t, 2>{ld, 1} : std::array<int64_t, 2>{1, ld};
  return torch::from_blob(const_cast<T *>(data), sizes, strides,
                          torch::TensorOptions().dtype(
                              c10::CppTypeToScalarType<T>::value));
}

// C := alpha * op(A) * op(B) + beta * C, op(A): M x K, op(B): K x N, C: M x N.
template <typename T>
void torch_gemm(const CBLAS_ORDER Layout, const CBLAS_TRANSPOSE TransA,
                const CBLAS_TRANSPOSE TransB, const int M, const int N,
                const int K, const T alpha, const T *A, const int lda,
                const T *B, const int ldb, const T beta, T *C, const int ldc) {
  if (M <= 0 || N <= 0)
    return;
  const bool row_major = Layout == CblasRowMajor;
  const bool trans_a = TransA != CblasNoTrans;
  const bool trans_b = TransB != CblasNoTrans;
  auto c = blas_matrix(C, M, N, ldc, row_major);
  if (K <= 0 || alpha == T(0)) {
    if (beta == T(0))
      c.zero_();
    else if (beta != T(1))
      c.mul_(beta);
    return;
  }
  auto a = blas_matrix(A, trans_a ? K : M, trans_a ? M : K, lda, row_major);
  auto b = blas_matrix(B, trans_b ? N : K, trans_b ? K : N, ldb, row_major);
  if (trans_a)
    a = a.t();
  if (trans_b)
    b = b.t();
  // addmm_ ignores the input of C when beta == 0 (no NaN/Inf propagation).
  c.addmm_(a, b, beta, alpha);
}

} // namespace

extern "C" {

void cblas_sgemm(const CBLAS_ORDER Layout, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const float alpha, const float *A, const int lda,
                 const float *B, const int ldb, const float beta, float *C,
                 const int ldc) {
  torch_gemm<float>(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
                    beta, C, ldc);
}

void cblas_dgemm(const CBLAS_ORDER Layout, const CBLAS_TRANSPOSE TransA,
                 const CBLAS_TRANSPOSE TransB, const int M, const int N,
                 const int K, const double alpha, const double *A, const int lda,
                 const double *B, const int ldb, const double beta, double *C,
                 const int ldc) {
  torch_gemm<double>(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
                     beta, C, ldc);
}

void cblas_saxpy(const int N, const float alpha, const float *X,
                 const int incX, float *Y, const int incY) {
  auto X_tensor = torch::from_blob(const_cast<float*>(X), {N}, torch::kFloat32);
  auto Y_tensor = torch::from_blob(Y, {N}, torch::kFloat32);
  Y_tensor.add_(X_tensor, alpha);
}

void cblas_daxpy(const int N, const double alpha, const double *X,
                 const int incX, double *Y, const int incY) {
  auto X_tensor = torch::from_blob(const_cast<double*>(X), {N}, torch::kFloat64);
  auto Y_tensor = torch::from_blob(Y, {N}, torch::kFloat64);
  Y_tensor.add_(X_tensor, alpha);
}

} // extern "C"
#endif // USE_MKL

namespace minkowski {

template <>
void cpu_gemm<float>(const CBLAS_ORDER Layout, const CBLAS_TRANSPOSE TransA,
                     const CBLAS_TRANSPOSE TransB, const int M, const int N,
                     const int K, const float alpha, const float *A,
                     const float *B, const float beta, float *C) {
  int lda, ldb, ldc;
  if (Layout == CblasRowMajor) {
    lda = (TransA == CblasNoTrans) ? K : M;
    ldb = (TransB == CblasNoTrans) ? N : K;
    ldc = N;
  } else {
    lda = (TransA == CblasNoTrans) ? M : K;
    ldb = (TransB == CblasNoTrans) ? K : N;
    ldc = M;
  }
  cblas_sgemm(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
              ldc);
}

template <>
void cpu_gemm<double>(const CBLAS_ORDER Layout, const CBLAS_TRANSPOSE TransA,
                      const CBLAS_TRANSPOSE TransB, const int M, const int N,
                      const int K, const double alpha, const double *A,
                      const double *B, const double beta, double *C) {
  int lda, ldb, ldc;
  if (Layout == CblasRowMajor) {
    lda = (TransA == CblasNoTrans) ? K : M;
    ldb = (TransB == CblasNoTrans) ? N : K;
    ldc = N;
  } else {
    lda = (TransA == CblasNoTrans) ? M : K;
    ldb = (TransB == CblasNoTrans) ? K : N;
    ldc = M;
  }
  cblas_dgemm(Layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C,
              ldc);
}

template <>
void cpu_add<float>(const int n, const float *a, const float *b, float *y) {
  vsAdd(n, a, b, y);
}

template <>
void cpu_add<double>(const int n, const double *a, const double *b, double *y) {
  vdAdd(n, a, b, y);
}

template <>
void cpu_mul<float>(const int n, const float *a, const float *b, float *y) {
  vsMul(n, a, b, y);
}

template <>
void cpu_mul<double>(const int n, const double *a, const double *b, double *y) {
  vdMul(n, a, b, y);
}

template <>
void cpu_div<float>(const int n, const float *a, const float *b, float *y) {
  vsDiv(n, a, b, y);
}

template <>
void cpu_div<double>(const int n, const double *a, const double *b, double *y) {
  vdDiv(n, a, b, y);
}

template <>
void cpu_axpy<float>(const int N, const float alpha, const float *X, float *Y) {
  cblas_saxpy(N, alpha, X, 1, Y, 1);
}

template <>
void cpu_axpy<double>(const int N, const double alpha, const double *X,
                      double *Y) {
  cblas_daxpy(N, alpha, X, 1, Y, 1);
}

} // end namespace minkowski
