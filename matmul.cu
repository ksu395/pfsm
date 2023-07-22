#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "matmul.h"


static const size_t blockSizeX = 16;
static const size_t blockSizeY = 16;


template <typename scalar_t>
__global__ void fused_matmul_cuda_kernel(
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> scores,
        const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> max_i,
        const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> sum_exp_i,
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> v,
        torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output)
{
    const auto a_cols = scores.size(1);
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    output[row][col] = 0.0;
    for (int i = 0; i < a_cols; ++i) {
        output[row][col] += exp(scores[row][i] - max_i[row]) * v[i][col] / sum_exp_i[row];
    }
}

torch::Tensor fused_matmul_cuda(
        const torch::Tensor scores,
        const torch::Tensor max_i,
        const torch::Tensor sum_exp_i,
        const torch::Tensor v)
{
    const auto a_rows = scores.size(0);
    const auto a_cols = scores.size(1);
    const auto b_rows = v.size(0);
    const auto b_cols = v.size(1);
    // for simplicity, enforce power of 2 shape
    assert(a_rows > 0 && !(a_rows & (a_rows-1)));
    assert(a_cols > 0 && !(a_cols & (a_cols-1)));
    assert(max_i.size(0) == a_rows);
    assert(sum_exp_i.size(0) == a_rows);
    assert(b_rows > 0 && !(b_rows & (b_rows-1)));
    assert(b_cols > 0 && !(b_cols & (b_cols-1)));
    assert(a_cols == b_rows);

    const dim3 dimBlock(blockSizeX, blockSizeY);
    const dim3 dimGrid(b_cols / dimBlock.x, a_rows / dimBlock.y);

    // todo: replace with cuda malloc?
    auto output = scores.new_empty({long(a_rows), long(b_cols)});

    AT_DISPATCH_FLOATING_TYPES(scores.type(), "reduce_cuda_kernel", ([&] {
        fused_matmul_cuda_kernel<scalar_t><<<dimGrid, dimBlock>>>(
                scores.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                max_i.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                sum_exp_i.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                v.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
    }));

    return output;
}
