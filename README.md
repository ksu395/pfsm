# PFSM: Partially Fused SoftMax

See paper [Speed Is All You Need: On-Device Acceleration of Large Diffusion Models via GPU-Aware Optimizations](https://arxiv.org/pdf/2304.11267.pdf), 
specifically section 3.2.1 and figure 2.
Additonal details can be found in the related [blog post](https://ai.googleblog.com/2023/06/speed-is-all-you-need-on-device.html).<br><br>
For simplicity. this code does not consider:
- backward functions
- batch > 1

Functionally, this code performs the following operations:<br>
```
scores = torch.nn.functional.softmax(qkt, dim=-1)
y = torch.matmul(scores, v)
```

The idea here is to never create the scores matrix, as it can be quite large (seq_len, seq_len).  Instead, we do the softmax reduction ops first, which have much smaller results, and then defer the softmax elementwise ops as part of matmul.


## Objective

The main objective here was to educate myself on somewhat more advanced CUDA kernel programming techniques. My prior experience in this area was rather basic.  This code has decent performance, but could be much more optimized.

As I got into the implementation, I realized that this fusion is probably not a good one for Ampere or later Nvidia cards, as using TensorCores is probably better.
I believe that TensorCores cannot be used in this fusion, as there are row-dependent math ops needed for matrix A.  
Per the docs, kernels that utilize TensorCores can only apply position-independent operations, e.g. scale or offset by a scalar.

If you are looking for better acceleration on modern Nvidia, please see [FlashAttention (fp16)](https://pytorch.org/blog/accelerated-pytorch-2/).

## PyTorch Attention (attn.py)

There are three different implementations using pytorch operations.  Note that this is not a full model, just a few consecutive functions that perform a single Scaled Dot-Product Attention, as described in [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) paper, section 3.2.1.  

### Pytorch SDP (gold_sdp_attn())

This is the reference implementation.  It simply calls torch.nn.functional.scaled_dot_product_attention() to provide golden output for testing purposes.  See the blog post [Accelerated PyTorch 2 Transformers](https://pytorch.org/blog/accelerated-pytorch-2/) for more details on the implementation.

### Pytorch PFSM (my_sdp_attn() + py_partially_fused_softmax())

This pytorch implementation a) decomposes the SDP into the individual oprations, and b) performs the PFSM.  The latter is computed by:
1. column reduction to compute row-wise max
2. column reduction to compute row-wise sum of exponentials
3. fusing the elementwise ops with Scores matrix during the matmul with Values matrix

This was a secondary reference implementation that provided intermediate vectors for debugging the CUDA implementation described in the next section.

### CUDA PFSM (my_sdp_attn() + cu_partially_fused_softmax())

This implementation does the same operations as the previous section, but delegates the math to CUDA via the Ninja JIT compiler and pybind linkage to the C code.  See [CUSTOM C++ AND CUDA EXTENSIONS](https://pytorch.org/tutorials/advanced/cpp_extension.html) for more details.

The two CUDA kernels are described in more detail in the following sections.

#### Reduction Ops (reduce.cu)

The initial implementation split the input in half horizontally and each thread reduced one pixel from left half with one pixel from the right half, and then recursively continued until only a single column of output remained.
Functionally, this worked fine, but the first partial output was still quite large (seq_len, seq_len / 2).
This is counter to the purpose of this fusion: to reduce intermediate tensor sizes.

While searching the Internet for a better solution, I came upon an Nvidia presentation by Mark Harris on [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf).
This deck has several advanced optimizations that I did not implement, but I did borrow the concept of using shared memory to collaborate within a thread block to reduce multiple columns in one kernel invocation.
This solved the problem above: the first partial output size is now only (seq_len, gridDim.x).
Plus, it probably runs much faster.

#### Fused Matmul (matmul.cu)

Again, the initial implementation was a very basic vector dot-product, with one thread per output pixel.

The CUDA C Programming guide, section [3.2.4 Shared Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory), describes how to use shared memory to reduce the number of memory accesses.  After working with shared memory in the reduce code, this was a fairly straight forward enhancement to implement.

## Addtional References
[An Even Easier Introduction to CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)  
[Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
