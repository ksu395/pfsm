#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


template <typename scalar_t>
__global__ void max_cuda_kernel(
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input,
        const size_t num_rows_in,
        torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
        const size_t num_cols_out)
{
    const int column = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < num_rows_in && column < num_cols_out)
        output[row][column] = fmax(input[row][column], input[row][column + num_cols_out]);
}

torch::Tensor partial_max_cuda(
        const torch::Tensor input,
        const size_t num_rows_in,
        const size_t num_cols_in)
{
    // this is larger than I like (half of input)
    // 'online' might help...or do more than 2 cols at a time?
    auto num_cols_out = num_cols_in / 2;
    auto output = input.new_empty({long(num_rows_in), long(num_cols_out)});

    const dim3 threadsPerBlock(16, 16);
    const dim3 blocks((num_cols_out + threadsPerBlock.x - 1) / threadsPerBlock.x,
                      (num_rows_in + threadsPerBlock.y - 1) / threadsPerBlock.y);

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_cuda_kernel", ([&] {
        max_cuda_kernel<scalar_t><<<blocks, threadsPerBlock>>>(
                    input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    num_rows_in,
                    output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    num_cols_out);
    }));

    if (num_cols_out > 1) {
        return partial_max_cuda(output, num_rows_in, num_cols_out);
    }

    return output;
}


torch::Tensor max_cuda(
        const torch::Tensor input,
        int dim)
{
    // for simplicity, only allow 2D reduction along rows
    assert(dim == -1);

    const auto num_rows = input.size(0);
    const auto num_cols = input.size(1);
    // for simplicity, enforce power of 2 shape
    assert(num_rows > 0 && !(num_rows & (num_rows-1)));
    assert(num_cols > 0 && !(num_cols & (num_cols-1)));

    auto output = partial_max_cuda(input, num_rows, num_cols);
    output.squeeze_();
    
    return output;
}