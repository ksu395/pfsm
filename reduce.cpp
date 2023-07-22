#include <torch/extension.h>

#include "reduce.h"

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor max(
        const torch::Tensor input,
        int dim)
{
    CHECK_INPUT(input);
    // for simplicity, only allow 2D reduction along rows
    assert(dim == -1);

    return reduce_max_cuda(input);
}

torch::Tensor sum_exp(
        const torch::Tensor input,
        const torch::Tensor max_i,
        const int dim)
{
    CHECK_INPUT(input);
    CHECK_INPUT(max_i);
    // for simplicity, only allow 2D reduction along rows
    assert(dim == -1);

    return reduce_sum_exp_cuda(input, max_i);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_exp", &sum_exp, "reduce sum of exponents (CUDA)");
    m.def("max", &max, "reduce max (CUDA)");
}

