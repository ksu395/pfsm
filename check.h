//
// Created by ubuntu on 7/23/23.
//

#ifndef PFSM_CHECK_H
#define PFSM_CHECK_H

#include <torch/extension.h>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_POW_OF_2(x) TORCH_CHECK(x > 0 && !(x & (x-1)), #x " must be a power of 2")
#define CHECK_EQUAL(x,y) TORCH_CHECK(x == y, #x " must equal " #y)

#endif //PFSM_CHECK_H
