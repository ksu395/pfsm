#include <torch/extension.h>

#include "matmul.h"


torch::Tensor fused_matmul(
        const torch::Tensor scores,
        const torch::Tensor max_i,
        const torch::Tensor sum_exp_i,
        const torch::Tensor v)
{
    CHECK_INPUT(scores);
    CHECK_INPUT(max_i);
    CHECK_INPUT(sum_exp_i);
    CHECK_INPUT(v);

    return fused_matmul_cuda(scores, max_i, sum_exp_i, v);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_matmul", &fused_matmul, "fused matmul (CUDA)");
}

