import torch
from torch.utils.cpp_extension import load


torch.manual_seed(13)

seq_len = 1024
embed_dim = 128

query = torch.randn(seq_len, embed_dim).cuda()
key = torch.randn(seq_len, embed_dim).cuda()
value = torch.randn(seq_len, embed_dim).cuda()

# disable tf32 since my implementation uses vector dot product, not matmul
# if enabled, need to allow for more tolerance in allclose
torch.backends.cuda.matmul.allow_tf32 = False
rtol, atol = 1e-04, 1e-07

use_gpu = True

my_reduce = load(name='my_reduce', sources=['reduce.cpp', 'reduce.cu'], verbose=True)
my_matmul = load(name='my_matmul', sources=['matmul.cpp', 'matmul.cu'], verbose=True)


# See paper "Speed Is All You Need: On-Device Acceleration of Large Diffusion Models via GPU-Aware Optimizations"
# Specifically section 3.2.1 and figure 2
# For simplicity. this code does not consider the needed backward function
# For simplicity. this code does not consider batch > 1

# Functionally, this code performs the following operations:
#
# scores = torch.nn.functional.softmax(qkt, dim=-1)
# y = torch.matmul(scores, v)
#
# the idea here is to never create the scores matrix, as it can be quite large (seq_len x seq_len)
# instead, we do the softmax reduction ops first, which have smaller results, and then defer
# the softmax elementwise ops as part of matmul

def cu_partially_fused_softmax(qkt: torch.tensor, v: torch.tensor) -> torch.tensor:

    # compute the max and denominator, one per row
    qkti_max = my_reduce.max(qkt, -1)
    sum_of_exp = my_reduce.sum_exp(qkt, qkti_max, -1)

    # compute the fused matmul
    y = my_matmul.fused_matmul(qkt, qkti_max, sum_of_exp, v)

    y2 = torch.empty_like(v)
    nrows, ncols = v.size()
    for i in range(nrows):
        # elementwise portion of softmax, one row of qkt
        qkti = torch.exp(qkt[i] - qkti_max[i]) / sum_of_exp[i]
        for j in range(ncols):
            # vector dot product with column of v
            y2[i][j] = torch.vdot(qkti, v[:, j]).item()
    assert(torch.allclose(y, y2, rtol=rtol, atol=atol))

    # all done :)
    return y


def py_partially_fused_softmax(qkt: torch.tensor, v: torch.tensor) -> torch.tensor:

    # compute the max and denominator, one per row
    nrows, ncols = v.size()
    qkti_max = qkt.max(-1)[0]  # ignore indices
    sum_of_exp = [torch.exp(qkt[i] - qkti_max[i]).sum().item() for i in range(nrows)]

    # compute the matmul, one output pixel at a time
    y = torch.empty_like(v)
    for i in range(nrows):
        # elementwise portion of softmax, one row of qkt
        qkti = torch.exp(qkt[i] - qkti_max[i]) / sum_of_exp[i]
        for j in range(ncols):
            # vector dot product with column of v
            y[i][j] = torch.vdot(qkti, v[:, j]).item()

    # all done :)
    return y


def my_attn(q: torch.tensor, k: torch.tensor, v: torch.tensor) -> torch.tensor:
    q = q / (embed_dim ** 0.5)
    qkt = torch.matmul(q, k.transpose(-2, -1))
    if use_gpu:
        y = cu_partially_fused_softmax(qkt, v)
    else:
        y = py_partially_fused_softmax(qkt, v)
    return y


# see "Attention Is All You Need"
def sdp_attn(q: torch.tensor, k: torch.tensor, v: torch.tensor) -> torch.tensor:
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


# compare reference implementation with ours to ensure they produce essentially equal results
with torch.inference_mode():
    attn_output_golden = sdp_attn(query, key, value)
    print(F'attn_output_golden:\n{attn_output_golden}')
    attn_output = my_attn(query, key, value)
    print(F'attn_output:\n{attn_output}')
    if not torch.allclose(attn_output_golden, attn_output, rtol=rtol, atol=atol):
        for row in range(attn_output_golden.size(-2)):
            if not torch.allclose(attn_output_golden[row, :], attn_output[row, :], rtol=rtol, atol=atol):
                for col in range(attn_output_golden.size(-1)):
                    if not torch.allclose(attn_output_golden[row, col], attn_output[row, col], rtol=rtol, atol=atol):
                        print(F'not close gold={attn_output_golden[row, col]}, mine={attn_output[row, col]}')
        assert False
