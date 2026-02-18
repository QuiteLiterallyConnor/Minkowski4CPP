
#ifndef MINK_CPP_AUTOGRAD_SPMM_FN_HPP
#define MINK_CPP_AUTOGRAD_SPMM_FN_HPP

#include <torch/torch.h>

namespace minkowski {

template <typename th_int_type>
torch::Tensor coo_spmm(torch::Tensor const &rows, torch::Tensor const &cols,
                       torch::Tensor const &vals, int64_t const dim_i,
                       int64_t const dim_j, torch::Tensor const &mat2,
                       int64_t const spmm_algorithm_id, bool const is_sorted);

template <typename th_int_type>
std::vector<torch::Tensor>
coo_spmm_average(torch::Tensor const &rows, torch::Tensor const &cols,
                 int64_t const dim_i, int64_t const dim_j,
                 torch::Tensor const &mat2, int64_t const spmm_algorithm_id);

inline torch::Tensor spmm_cpu(torch::Tensor const &rows,
                              torch::Tensor const &cols,
                              torch::Tensor const &vals,
                              int64_t dim_i, int64_t dim_j,
                              torch::Tensor const &mat) {
  auto COO = torch::stack({rows.to(torch::kLong), cols.to(torch::kLong)}, 0);
  auto sp = torch::sparse_coo_tensor(COO, vals, {dim_i, dim_j},
                                     mat.options().layout(torch::kSparse));
  return sp.matmul(mat);
}

inline std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
spmm_average_cpu(torch::Tensor const &rows, torch::Tensor const &cols,
                 int64_t dim_i, int64_t dim_j,
                 torch::Tensor const &mat) {
  auto rows_sorted = rows.clone();
  auto cols_sorted = cols.clone();
  auto sort_ind = std::get<1>(rows_sorted.sort(0));
  rows_sorted = rows.index_select(0, sort_ind);
  cols_sorted = cols.index_select(0, sort_ind);

  auto COO = torch::stack({rows_sorted.to(torch::kLong),
                           cols_sorted.to(torch::kLong)}, 0);

  auto unique_result = at::_unique2(rows_sorted, true,
                                    true,
                                    true);
  auto inverse_ind = std::get<1>(unique_result);
  auto counts = std::get<2>(unique_result).to(torch::kFloat);
  auto vals = (1.0f / counts.index_select(0, inverse_ind)).to(mat.dtype());

  auto sp = torch::sparse_coo_tensor(COO, vals, {dim_i, dim_j},
                                     mat.options().layout(torch::kSparse));
  auto result = sp.matmul(mat);
  return {result, COO, vals};
}

inline torch::Tensor spmm_dispatch(torch::Tensor const &rows,
                                   torch::Tensor const &cols,
                                   torch::Tensor const &vals,
                                   int64_t dim_i, int64_t dim_j,
                                   torch::Tensor const &mat,
                                   int64_t cuda_spmm_alg,
                                   bool is_sorted) {
  if (mat.is_cuda()) {
#ifdef CPU_ONLY
    TORCH_CHECK(false, "CUDA not available in CPU_ONLY build");
    return {};
#else
    return coo_spmm<int32_t>(rows.to(torch::kInt), cols.to(torch::kInt),
                             vals, dim_i, dim_j, mat, cuda_spmm_alg,
                             is_sorted);
#endif
  } else {
    return spmm_cpu(rows, cols, vals, dim_i, dim_j, mat);
  }
}

struct SPMMFunction
    : public torch::autograd::Function<SPMMFunction> {

  static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                            at::Tensor rows,
                            at::Tensor cols,
                            at::Tensor vals,
                            int64_t dim_i,
                            int64_t dim_j,
                            at::Tensor mat,
                            int64_t cuda_spmm_alg = 1);

  static torch::autograd::variable_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::variable_list grad_outputs);
};

struct SPMMAverageFunction
    : public torch::autograd::Function<SPMMAverageFunction> {

  static at::Tensor forward(torch::autograd::AutogradContext *ctx,
                            at::Tensor rows,
                            at::Tensor cols,
                            int64_t dim_i,
                            int64_t dim_j,
                            at::Tensor mat,
                            int64_t cuda_spmm_alg = 1);

  static torch::autograd::variable_list
  backward(torch::autograd::AutogradContext *ctx,
           torch::autograd::variable_list grad_outputs);
};

} 

#endif 
