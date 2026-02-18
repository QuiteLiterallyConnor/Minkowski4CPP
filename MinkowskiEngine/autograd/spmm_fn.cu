
#include "spmm_fn.hpp"

namespace minkowski {

at::Tensor SPMMFunction::forward(
    torch::autograd::AutogradContext *ctx,
    at::Tensor rows,
    at::Tensor cols,
    at::Tensor vals,
    int64_t dim_i,
    int64_t dim_j,
    at::Tensor mat,
    int64_t cuda_spmm_alg) {

  ctx->saved_data["dim_i"] = dim_i;
  ctx->saved_data["dim_j"] = dim_j;
  ctx->saved_data["cuda_spmm_alg"] = cuda_spmm_alg;
  ctx->save_for_backward({rows, cols, vals});

  auto result = spmm_dispatch(rows, cols, vals, dim_i, dim_j, mat,
                              cuda_spmm_alg, false);
  return result;
}

torch::autograd::variable_list SPMMFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_outputs) {

  auto saved = ctx->get_saved_variables();
  auto rows = saved[0];
  auto cols = saved[1];
  auto vals = saved[2];

  auto dim_i = ctx->saved_data["dim_i"].toInt();
  auto dim_j = ctx->saved_data["dim_j"].toInt();
  auto cuda_spmm_alg = ctx->saved_data["cuda_spmm_alg"].toInt();

  auto grad = grad_outputs[0].contiguous();

  auto grad_mat = spmm_dispatch(cols, rows, vals, dim_j, dim_i, grad,
                                cuda_spmm_alg, false);

  return {at::Tensor(), at::Tensor(), at::Tensor(),
          at::Tensor(), at::Tensor(), grad_mat, at::Tensor()};
}

at::Tensor SPMMAverageFunction::forward(
    torch::autograd::AutogradContext *ctx,
    at::Tensor rows,
    at::Tensor cols,
    int64_t dim_i,
    int64_t dim_j,
    at::Tensor mat,
    int64_t cuda_spmm_alg) {

  ctx->saved_data["dim_i"] = dim_i;
  ctx->saved_data["dim_j"] = dim_j;
  ctx->saved_data["cuda_spmm_alg"] = cuda_spmm_alg;

  at::Tensor result, COO, vals;

  if (mat.is_cuda()) {
#ifdef CPU_ONLY
    TORCH_CHECK(false, "CUDA not available in CPU_ONLY build");
#else
    auto gpu_result = coo_spmm_average<int32_t>(
        rows.to(torch::kInt), cols.to(torch::kInt),
        dim_i, dim_j, mat, cuda_spmm_alg);
    TORCH_CHECK(gpu_result.size() == 3,
                "coo_spmm_average should return 3 tensors");
    result = gpu_result[0];
    COO = gpu_result[1];
    vals = gpu_result[2];
#endif
  } else {
    auto cpu_result = spmm_average_cpu(rows, cols, dim_i, dim_j, mat);
    result = std::get<0>(cpu_result);
    COO = std::get<1>(cpu_result);
    vals = std::get<2>(cpu_result);
  }

  ctx->save_for_backward({COO, vals});
  return result;
}

torch::autograd::variable_list SPMMAverageFunction::backward(
    torch::autograd::AutogradContext *ctx,
    torch::autograd::variable_list grad_outputs) {

  auto saved = ctx->get_saved_variables();
  auto COO = saved[0];
  auto vals = saved[1];

  auto dim_i = ctx->saved_data["dim_i"].toInt();
  auto dim_j = ctx->saved_data["dim_j"].toInt();
  auto cuda_spmm_alg = ctx->saved_data["cuda_spmm_alg"].toInt();

  auto grad = grad_outputs[0].contiguous();

  auto grad_mat = spmm_dispatch(COO[1], COO[0], vals, dim_j, dim_i, grad,
                                cuda_spmm_alg, false);

  return {at::Tensor(), at::Tensor(), at::Tensor(),
          at::Tensor(), grad_mat, at::Tensor()};
}

} 
