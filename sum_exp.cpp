#include <torch/extension.h>

#include <vector>

// CUDA forward declaration
torch::Tensor sum_exp_cuda(
        torch::Tensor input,
        torch::Tensor max_i);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor sum_exp(
        const torch::Tensor input,
        const torch::Tensor max_i)
{
    CHECK_INPUT(input);

    return sum_exp_cuda(input, max_i);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_exp", &sum_exp, "sum_exp (CUDA)");
}
