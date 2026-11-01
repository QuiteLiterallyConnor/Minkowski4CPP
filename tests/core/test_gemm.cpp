// CPU BLAS shim (src/math_functions_cpu.cpp) against torch::mm for every layout/transpose
// combination, non-square sizes, padded leading dimensions, alpha/beta handling.
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "src/math_functions.hpp"
#include "utils/test_framework.hpp"

namespace {

template <typename T>
constexpr c10::ScalarType dtype_of() {
    return std::is_same_v<T, float> ? torch::kFloat32 : torch::kFloat64;
}

// Writes the logical matrix S (rows x cols) into a BLAS buffer with leading dimension ld.
template <typename T>
std::vector<T> store(const at::Tensor& S, bool row_major, int ld, T fill) {
    const int64_t rows = S.size(0), cols = S.size(1);
    std::vector<T> buf((row_major ? rows : cols) * ld, fill);
    auto s = S.to(dtype_of<T>()).contiguous();
    const T* p = s.data_ptr<T>();
    for (int64_t i = 0; i < rows; ++i)
        for (int64_t j = 0; j < cols; ++j)
            buf[row_major ? i * ld + j : i + j * ld] = p[i * cols + j];
    return buf;
}

template <typename T>
at::Tensor load(const std::vector<T>& buf, int64_t rows, int64_t cols, bool row_major, int ld) {
    auto out = torch::empty({rows, cols}, torch::kFloat64);
    double* p = out.data_ptr<double>();
    for (int64_t i = 0; i < rows; ++i)
        for (int64_t j = 0; j < cols; ++j)
            p[i * cols + j] = buf[row_major ? i * ld + j : i + j * ld];
    return out;
}

template <typename T>
void call_gemm(bool via_cblas, CBLAS_ORDER layout, CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, int M,
               int N, int K, T alpha, const T* A, int lda, const T* B, int ldb, T beta, T* C,
               int ldc) {
    if (!via_cblas) {
        minkowski::cpu_gemm<T>(layout, ta, tb, M, N, K, alpha, A, B, beta, C);
    } else if constexpr (std::is_same_v<T, float>) {
        cblas_sgemm(layout, ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    } else {
        cblas_dgemm(layout, ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }
}

// C := alpha * op(A) * op(B) + beta * C with op(A): M x K, op(B): K x N, C: M x N.
// `pad` > 0 adds unused columns/rows to every leading dimension (only meaningful via cblas).
template <typename T>
void check_gemm(CBLAS_ORDER layout, CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, int M, int N, int K,
                T alpha, T beta, int pad, bool via_cblas, uint64_t seed) {
    torch::manual_seed(seed);
    const bool rm = layout == CblasRowMajor;
    const bool tA = ta != CblasNoTrans, tB = tb != CblasNoTrans;
    auto opA = torch::randn({M, K}, torch::kFloat64);
    auto opB = torch::randn({K, N}, torch::kFloat64);
    auto C0 = torch::randn({M, N}, torch::kFloat64);
    auto SA = tA ? opA.t() : opA;  // stored matrices
    auto SB = tB ? opB.t() : opB;
    const int lda = int(rm ? SA.size(1) : SA.size(0)) + pad;
    const int ldb = int(rm ? SB.size(1) : SB.size(0)) + pad;
    const int ldc = (rm ? N : M) + pad;
    const T sentinel = T(12345);
    auto a = store<T>(SA, rm, lda, sentinel);
    auto b = store<T>(SB, rm, ldb, sentinel);
    // beta == 0 must not read C (BLAS semantics): fill it with NaN.
    const T c_init = beta == T(0) ? std::numeric_limits<T>::quiet_NaN() : T(0);
    auto c = store<T>(beta == T(0) ? torch::full({M, N}, double(c_init), torch::kFloat64) : C0,
                      rm, ldc, sentinel);

    call_gemm<T>(via_cblas, layout, ta, tb, M, N, K, alpha, a.data(), lda, b.data(), ldb, beta,
                 c.data(), ldc);

    auto expected = double(alpha) * opA.mm(opB);
    if (beta != T(0)) expected += double(beta) * C0;
    auto got = load<T>(c, M, N, rm, ldc);
    const double tol = std::is_same_v<T, float> ? 1e-4 : 1e-12;
    if (!torch::allclose(got, expected, tol, tol))
        FAIL_AT("gemm mismatch: " << (rm ? "RowMajor" : "ColMajor") << " " << (tA ? "T" : "N")
                                  << (tB ? "T" : "N") << " M=" << M << " N=" << N << " K=" << K
                                  << " alpha=" << alpha << " beta=" << beta << " pad=" << pad
                                  << (via_cblas ? " (cblas)" : " (cpu_gemm)") << " max diff "
                                  << (got - expected).abs().max().item<double>());
    // Padding of C must stay untouched.
    for (int64_t i = 0; i < int64_t(c.size()); ++i) {
        // Index within one leading-dimension stripe (a row if RowMajor, a column if ColMajor).
        if (i % ldc >= (rm ? N : M) && c[i] != sentinel)
            FAIL_AT("gemm wrote into the padding of C at " << i);
    }
}

template <typename T>
void check_all_combinations(bool via_cblas, int pad) {
    const int sizes[][3] = {{5, 7, 3}, {4, 2, 6}, {1, 9, 4}, {8, 3, 1}, {6, 6, 6}};
    uint64_t seed = 1;
    for (auto layout : {CblasRowMajor, CblasColMajor})
        for (auto ta : {CblasNoTrans, CblasTrans})
            for (auto tb : {CblasNoTrans, CblasTrans})
                for (auto& s : sizes) {
                    check_gemm<T>(layout, ta, tb, s[0], s[1], s[2], T(1), T(0), pad, via_cblas,
                                  seed++);
                    check_gemm<T>(layout, ta, tb, s[0], s[1], s[2], T(0.5), T(1), pad, via_cblas,
                                  seed++);
                    check_gemm<T>(layout, ta, tb, s[0], s[1], s[2], T(-2), T(0.25), pad,
                                  via_cblas, seed++);
                }
}

}  // namespace

TEST(gemm_cpu_gemm_float_all_layouts_and_transposes) { check_all_combinations<float>(false, 0); }

TEST(gemm_cpu_gemm_double_all_layouts_and_transposes) { check_all_combinations<double>(false, 0); }

TEST(gemm_cblas_sgemm_padded_leading_dimensions) { check_all_combinations<float>(true, 3); }

TEST(gemm_cblas_dgemm_padded_leading_dimensions) { check_all_combinations<double>(true, 3); }

TEST(gemm_cpu_div_double_divides) {
    std::vector<double> a = {1.0, 6.0, -9.0}, b = {2.0, 3.0, 4.5}, y(3);
    minkowski::cpu_div<double>(3, a.data(), b.data(), y.data());
    CHECK_NEAR(y[0], 0.5, 1e-15);
    CHECK_NEAR(y[1], 2.0, 1e-15);
    CHECK_NEAR(y[2], -2.0, 1e-15);
    std::vector<float> af = {1.f, 6.f}, bf = {2.f, 3.f}, yf(2);
    minkowski::cpu_div<float>(2, af.data(), bf.data(), yf.data());
    CHECK_NEAR(yf[0], 0.5, 1e-7);
    CHECK_NEAR(yf[1], 2.0, 1e-7);
}
