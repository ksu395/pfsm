import torch
from torch.utils.cpp_extension import load
import timeit


torch.manual_seed(13)

seq_len = 1024
embed_dim = 256

query = torch.randn(seq_len, embed_dim).cuda()
key = torch.randn(seq_len, embed_dim).cuda()
value = torch.randn(seq_len, embed_dim).cuda()

rtol, atol = 1e-04, 1e-06

use_gpu = True

my_reduce = load(name='my_reduce', sources=['reduce.cpp', 'reduce.cu'], verbose=True)
my_matmul = load(name='my_matmul', sources=['matmul.cpp', 'matmul.cu'], verbose=True)


# See READMD and "Speed Is All You Need"
# my pfsm
def cu_partially_fused_softmax(qkt: torch.tensor, v: torch.tensor) -> torch.tensor:

    # compute the max and denominator, one per row
    qkti_max = my_reduce.max(qkt, -1)
    sum_of_exp = my_reduce.sum_exp(qkt, qkti_max, -1)

    # compute the fused matmul
    y = my_matmul.fused_matmul(qkt, qkti_max, sum_of_exp, v)

    # all done :)
    return y


# reference pfsm
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


def my_sdp_attn(q: torch.tensor, k: torch.tensor, v: torch.tensor) -> torch.tensor:
    # first matmul is not part of the fusion
    q = q / (embed_dim ** 0.5)
    qkt = torch.matmul(q, k.transpose(-2, -1))
    if use_gpu:
        y = cu_partially_fused_softmax(qkt, v)
    else:
        y = py_partially_fused_softmax(qkt, v)
    return y


# see README and "Attention Is All You Need"
def gold_sdp_attn(q: torch.tensor, k: torch.tensor, v: torch.tensor) -> torch.tensor:
    return torch.nn.functional.scaled_dot_product_attention(q, k, v)


# compare reference implementation with ours to ensure they produce essentially equal results
with torch.inference_mode():
    # disable tf32 since my implementations use vector dot product, not matmul
    # if enabled, need to allow for more tolerance in allclose
    torch.backends.cuda.matmul.allow_tf32 = False
    attn_output_golden = gold_sdp_attn(query, key, value)
    print(F'attn_output_golden:\n{attn_output_golden}')
    attn_output = my_sdp_attn(query, key, value)
    print(F'attn_output:\n{attn_output}')
    if not torch.allclose(attn_output_golden, attn_output, rtol=rtol, atol=atol):
        for row in range(attn_output_golden.size(-2)):
            if not torch.allclose(attn_output_golden[row, :], attn_output[row, :], rtol=rtol, atol=atol):
                for col in range(attn_output_golden.size(-1)):
                    if not torch.allclose(attn_output_golden[row, col], attn_output[row, col], rtol=rtol, atol=atol):
                        print(F'not close gold={attn_output_golden[row, col]}, mine={attn_output[row, col]}')
        assert False

    # simple elapsed time comparison (varies greatly with {seq_len,embed_dim})
    num_reps = 10000
    torch.backends.cuda.matmul.allow_tf32 = True
    tgt = timeit.timeit("gold_sdp_attn(query, key, value)", number=num_reps, globals=globals())
    torch.backends.cuda.matmul.allow_tf32 = False
    tgf = timeit.timeit("gold_sdp_attn(query, key, value)", number=num_reps, globals=globals())
    tm = timeit.timeit("my_sdp_attn(query, key, value)", number=num_reps, globals=globals())
    print(F'gold_tf32={tgt}, gold={tgf}, mine={tm}')
    # example:
    # seq_len = 1024
    # embed_dim = 256
    # gold_tf32=0.7734472790000382, gold=1.114761440000052, mine=6.112324146999981