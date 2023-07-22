# PFSM: Partially Fused SoftMax

See paper "Speed Is All You Need: On-Device Acceleration of Large Diffusion Models via GPU-Aware Optimizations", 
specifically section 3.2.1 and figure 2.<br><br>
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

The main objective here was to educate myself on somewhat more advanced CUDA kernel programming techniques.  
My prior experience in this area was rather basic.

As I got into the implementation, I realized that this fusion is probably not a good one for Ampere or later Nvidia cards, as using TensorCores is probably better.
I believe that TensorCores cannot be used in this fusion, as there are row-dependent math ops needed for matrix A.  
Per the docs, kernels that utilize TensorCores can only apply position-independent operations, e.g. scale or offset by a scalar.


## Reduction Ops

The initial implementation split the input in half and each thread reduced one pixel from left half with one pixel from the right half, and then recursively continued until only a single column of output remained.
Functionally, this worked fine, but the first partial output is still quite large (seq_len, seq_len / 2).
This is counter to the purpose of this fusion: to reduce intermediate tensor sizes.

While searching the Internet for a better solution, I came upon an Nvidia presentation by Mark Harris on "Optimizing Parallel Reduction in CUDA".
This deck has several advanced optimizations that I did not implement, but I did borrow the concept of using shared memory to collaborate within a thread block to reduce multiple columns in one kernel invocation.
This solved the problem above: the first partial output size is now only (seq_len, gridDim.x).
Plus, it probably runs much faster.

## Fused Matmul

Again, the initial implementation was a very basic vector dot-product, with one thread per output pixel.

The CUDA C Programming guide, section 3.2.4. Shared Memory, describes a way to use shared memory to reduce the number of memory accesses.  After working with shared memory in the reduce code, this was a fairly straight forward enhancement to implement.
 
