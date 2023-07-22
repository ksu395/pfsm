#ifndef PFSM_REDUCE_H
#define PFSM_REDUCE_H

#include <torch/extension.h>


extern torch::Tensor reduce_max_cuda(
        const torch::Tensor input);

extern torch::Tensor reduce_sum_exp_cuda(
        const torch::Tensor input,
        const torch::Tensor max_i);


#endif //PFSM_REDUCE_H
