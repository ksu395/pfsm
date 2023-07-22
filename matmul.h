#ifndef PFSM_MATMUL_H
#define PFSM_MATMUL_H

#include <torch/extension.h>


extern torch::Tensor fused_matmul_cuda(
        const torch::Tensor scores,
        const torch::Tensor max_i,
        const torch::Tensor sum_exp_i,
        const torch::Tensor v);


#endif //PFSM_MATMUL_H
