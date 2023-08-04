#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "reduce.h"


typedef enum {
    REDUCE_OP_MAX,
    REDUCE_OP_SUM_EXP,
    REDUCE_OP_SUM
} reduction_op;


// this should be the smallest of a) max TB size for the target and b) seq_len
// the larger it is, the smaller the size of first partial output
// but any larger than seq_len will result in idle threads
static const size_t blockSize = 512;


template <typename scalar_t, reduction_op reduce_op>
__global__ void reduce_cuda_kernel(
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input,
        const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> max_i,
        const size_t num_rows_in,
        const size_t num_cols_in,
        torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output)
{
    // we have half as many blocks as inputs in X, so process every other block
    const auto col_in = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    const auto row_in = blockIdx.y * blockDim.y + threadIdx.y;
    // this block will start the reduction with the adjacent memory (one block width away)
    auto stride = blockDim.x;

    if (row_in < num_rows_in && (col_in+stride) < num_cols_in) {
        // scratchpad for building up partial outputs
        // needs to be declared 1D, as the aspect ratio of the blocks changes between iterations
        __shared__ char* sp[blockSize*sizeof(scalar_t)];
        // this kernel only operates within a single row
        auto row_base_sp = reinterpret_cast<scalar_t*>(&sp[threadIdx.y*blockDim.x*sizeof(scalar_t)]);

        // initial value for scratchpad: reduction of this block and adjacent block
        if (reduce_op == REDUCE_OP_MAX) {
            row_base_sp[threadIdx.x] = max(input[row_in][col_in], input[row_in][col_in + stride]);
        }
        if (reduce_op == REDUCE_OP_SUM_EXP) {
            row_base_sp[threadIdx.x] = exp(input[row_in][col_in] - max_i[row_in]) +
                                    exp(input[row_in][col_in + stride] - max_i[row_in]);
        }
        if (reduce_op == REDUCE_OP_SUM) {
            row_base_sp[threadIdx.x] = input[row_in][col_in] + input[row_in][col_in + stride];
        }
        __syncthreads();

        // reduce the scratchpad via binary tree
        while (stride > 1) {
            stride /= 2;
            if (threadIdx.x < stride) {
                if (reduce_op == REDUCE_OP_MAX) {
                    row_base_sp[threadIdx.x] = max(row_base_sp[threadIdx.x], row_base_sp[threadIdx.x + stride]);
                }
                if (reduce_op == REDUCE_OP_SUM_EXP || reduce_op == REDUCE_OP_SUM) {
                    // EXP only applies to very first reduction above
                    row_base_sp[threadIdx.x] = row_base_sp[threadIdx.x] + row_base_sp[threadIdx.x + stride];
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            output[row_in][blockIdx.x] = row_base_sp[0];
        }
    }
}


template <reduction_op reduce_op>
torch::Tensor partial_reduce_cuda(
        const torch::Tensor input,
        const torch::Tensor max_i,
        const size_t num_rows_in,
        const size_t num_cols_in)
{
    // To start, blocks are as wide as possible to maximize reduction factor and thus
    // minimize temp storage for partial outputs.  In later iterations, num_cols_in will get
    // very small and blocks will get taller.  Also, blocks only need to cover half of the
    // input columns, as each thread can do the first partial reduction on the input.
    auto blockSizeX = min(blockSize, num_cols_in/2);
    auto blockSizeY = min(blockSize / blockSizeX, num_rows_in);
    const dim3 dimBlock(blockSizeX, blockSizeY);
    const dim3 dimGrid((num_cols_in/2) / dimBlock.x, num_rows_in / dimBlock.y);
    auto num_cols_out = dimGrid.x;

    // if the output is too large, the algo could need to change to an 'online' method
    auto output = input.new_empty({long(num_rows_in), long(num_cols_out)});

    AT_DISPATCH_FLOATING_TYPES(input.type(), "reduce_cuda_kernel", ([&] {
        reduce_cuda_kernel<scalar_t, reduce_op><<<dimGrid, dimBlock>>>(
                input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                max_i.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                num_rows_in,
                num_cols_in,
                output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>());
    }));

    // recurse until fully reduced
    if (num_cols_out > 1) {
        if (reduce_op == REDUCE_OP_SUM_EXP) {
            // EXP only applies to first iteration
            return partial_reduce_cuda<REDUCE_OP_SUM>(output, max_i, num_rows_in, num_cols_out);
        } else {
            return partial_reduce_cuda<reduce_op>(output, max_i, num_rows_in, num_cols_out);
        }
    }

    return output.squeeze();
}

torch::Tensor reduce_sum_exp_cuda(
        const torch::Tensor input,
        const torch::Tensor max_i)
{
    const auto num_rows = input.size(0);
    const auto num_cols = input.size(1);
    // for simplicity, enforce power of 2 shape
    CHECK_POW_OF_2(num_rows);
    CHECK_POW_OF_2(num_cols);
    CHECK_EQUAL(max_i.size(0), num_rows);

    return partial_reduce_cuda<REDUCE_OP_SUM_EXP>(input, max_i, num_rows, num_cols);
}

torch::Tensor reduce_max_cuda(
        const torch::Tensor input)
{
    const auto num_rows = input.size(0);
    const auto num_cols = input.size(1);
    // for simplicity, enforce power of 2 shape
    CHECK_POW_OF_2(num_rows);
    CHECK_POW_OF_2(num_cols);

    static torch::Tensor max_i = torch::empty({1}); // unused for REDUCE_OP_MAX
    return partial_reduce_cuda<REDUCE_OP_MAX>(input, max_i, num_rows, num_cols);
}