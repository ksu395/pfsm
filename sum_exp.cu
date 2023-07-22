#include "reduce.h"

torch::Tensor sum_exp_cuda(
        const torch::Tensor input,
        const torch::Tensor max_i)
{
    return reduce_cuda(input, max_i, REDUCE_OP_SUM_EXP);
}