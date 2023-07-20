#include <torch/extension.h>

#include <vector>

// CUDA forward declaration
torch::Tensor max_cuda(
        torch::Tensor input,
        int dim);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor max(
        const torch::Tensor input,
        int dim)
{
    CHECK_INPUT(input);

    return max_cuda(input, dim);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max", &max, "Max (CUDA)");
}
