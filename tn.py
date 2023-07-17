import torch
import numpy as np
import torchtext

torch.manual_seed(13)

seq_len = 128
embed_dim = 100

query = torch.randn(seq_len, embed_dim)
key = torch.randn(seq_len, embed_dim)
value = torch.randn(seq_len, embed_dim)


# See paper "Speed Is All You Need: On-Device Acceleration of Large Diffusion Models via GPU-Aware Optimizations"
# Specifically section 3.2.1 and figure 2
# For simplicity. this code does not consider the needed backward function
# For simplicity. this code does not consider batch > 1
def partially_fused_softmax(qkt: torch.tensor, v: torch.tensor) -> torch.tensor:
    # functionally, this code performs the following operations:
    #
    # scores = torch.nn.functional.softmax(qkt, dim=-1)
    # y = torch.matmul(scores, v)
    #
    # the idea here is to never create the scores matrix, as it is quite large
    # instead, we do the softmax reduction ops first, which have smaller results,
    # and then defer the softmax elementwise ops as part of matmul

    # Note: this code operates row by row, but it could easily be done in groups of rows if that
    # better fits the parallelization characteristics of the hardware, or tiles that align
    # with the size needed for a local gemm engine

    # compute the denominator, one per row
    nrows, ncols = qkt.size(-2), v.size(-1)
    qkti_max = qkt.max(-1)[0]  # ignore indices
    sum_of_exp = [np.exp(qkt[i] - qkti_max[i]).sum().item() for i in range(nrows)]

    # compute the matmul, one output pixel at a time (easily parallelized)
    y = torch.empty(nrows, ncols)
    for i in range(nrows):
        # elementwise portion of softmax, one row of qkt
        qkti = np.exp(qkt[i] - qkti_max[i]) / sum_of_exp[i]
        for j in range(ncols):
            # vector dot product with column of v
            y[i][j] = np.vdot(qkti, v[:, j]).item()

    # all done :)
    return y


def my_sdp_attn(q: torch.tensor, k: torch.tensor, v: torch.tensor) -> torch.tensor:
    q = q / np.sqrt(embed_dim)
    qkt = torch.matmul(q, k.transpose(-2, -1))
    return partially_fused_softmax(qkt, v)


# see "Attention Is All You Need"
def gold_sdp_attn(q: torch.tensor, k: torch.tensor, v: torch.tensor) -> torch.tensor:
    # SDP assumes a batch dimension, but we don't need it for this exercise
    sdp_attn = torchtext.nn.ScaledDotProduct(batch_first=True)
    y, _ = sdp_attn(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0))
    return y.squeeze(0)


# compare reference implementation with ours to ensure they produce essentially equal results
with torch.inference_mode():
    attn_output_golden = gold_sdp_attn(query, key, value)
    print(F'attn_output_golden:\n{attn_output_golden}')
    attn_output = my_sdp_attn(query, key, value)
    print(F'attn_output:\n{attn_output}')
    rtol, atol = 1e-05, 1e-07
    if not np.allclose(attn_output_golden, attn_output, rtol=rtol, atol=atol):
        for row in range(attn_output_golden.size(-2)):
            if not np.allclose(attn_output_golden[row, :], attn_output[row, :], rtol=rtol, atol=atol):
                for col in range(attn_output_golden.size(-1)):
                    if not np.allclose(attn_output_golden[row, col], attn_output[row, col], rtol=rtol, atol=atol):
                        print(F'not close gold={attn_output_golden[row, col]}, mine={attn_output[row, col]}')
        assert False



