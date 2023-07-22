#include "reduce.h"


torch::Tensor max_cuda(
        const torch::Tensor input,
        const int dim)
{
    // for simplicity, only allow 2D reduction along rows
    assert(dim == -1);

    torch::Tensor max_i = torch::empty({input.size(0)}); // unused for REDUCE_OP_MAX
    auto output = reduce_cuda(input, max_i, REDUCE_OP_MAX);

    return output;
}