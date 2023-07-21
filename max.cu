#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


// todo: increase to e.g. 512 when functional
static const size_t blockSize = 64;


template <typename scalar_t>
__global__ void max_cuda_kernel(
        const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input,
        const size_t num_rows_in,
        const size_t num_cols_in,
        torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output,
        const size_t num_cols_out)
{
    // we have half as many blocks as inputs in X, so process every other block
    const int col_in = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    const int row_in = blockIdx.y * blockDim.y + threadIdx.y;
    // this block will start the reduction with the adjacent memory (one block width away)
    size_t stride = blockDim.x;

    if (row_in < num_rows_in && (col_in+stride) < num_cols_in) {
        // scratchpad for building up partial outputs
        // needs to be declared 1D, as the aspect ratio of the blocks changes between iterations
        __shared__ char* sp[blockSize*sizeof(scalar_t)];
        // this kernel only works within a single row
        scalar_t* row_base = (scalar_t*)&sp[threadIdx.y*blockDim.x*sizeof(scalar_t)];

        // initial value for scratchpad: max of this block and adjacent block
        row_base[threadIdx.x] = fmax(input[row_in][col_in], input[row_in][col_in + stride]);
        __syncthreads();

        // reduce the scratchpad via binary tree
        while (stride > 1) {
            stride /= 2;
            if (threadIdx.x < stride)
                row_base[threadIdx.x] = fmax(row_base[threadIdx.x], row_base[threadIdx.x + stride]);
            __syncthreads();
      }

        if (threadIdx.x == 0)
            output[row_in][blockIdx.x] = row_base[0];
    }
}

torch::Tensor partial_max_cuda(
        const torch::Tensor input,
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
    // todo: is it ok to cover entire output with one grid, or should this use a smaller grid + stride pattern?
    const dim3 dimGrid(num_cols_in/2 / dimBlock.x, num_rows_in / dimBlock.y);
    auto num_cols_out = dimGrid.x;

    // if the output is too large, the algo could need to change to an 'online' method
    auto output = input.new_empty({long(num_rows_in), long(num_cols_out)});

    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_cuda_kernel", ([&] {
        max_cuda_kernel<scalar_t><<<dimGrid, dimBlock>>>(
                    input.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
                    num_rows_in,
                    num_cols_in,
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
        const int dim)
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