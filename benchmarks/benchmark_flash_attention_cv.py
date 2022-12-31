from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from flash_attn.utils.benchmark import benchmark_all, benchmark_forward, benchmark_backward, benchmark_combined
from flash_attn.bert_padding import unpad_input, pad_input
from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func


def attention_ref(qkv, dropout_p, upcast=False, causal=False):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        attn_mask: (batch_size, seqlen)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
        attention: softmax after dropout
    """
    q, k, v = (qkv.float() if upcast else qkv).unbind(dim=2)
    seqlen = qkv.shape[1]
    d = qkv.shape[-1]
    scores = torch.einsum('bthd,bshd->bhts', q, k / math.sqrt(d))
    # scores.masked_fill_(rearrange(~attn_mask, 'b s -> b 1 1 s'), float('-inf'))
    if causal:
        causal_mask = torch.triu(torch.ones(seqlen, seqlen, dtype=torch.bool, device=qkv.device), 1)
        scores.masked_fill_(causal_mask, float('-inf'))
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    # return output.to(dtype=qkv.dtype), attention.to(dtype=qkv.dtype)
    return output.to(dtype=qkv.dtype)


torch.manual_seed(0)
repeats = 1
batch_size = 2
nheads = 8
seqlen = 4096
n = 320
d = n // nheads
dropout_p = 0.0
causal = False
dtype = torch.float16
device = 'cuda'

x = torch.randn(batch_size, seqlen, n, device='cuda', dtype=dtype, requires_grad=True)
Wqkv = torch.nn.Linear(nheads * d, 3 * nheads * d, device=device, dtype=dtype)

lengths = torch.randint(seqlen+1, seqlen+2, (batch_size, 1), device='cuda')
attention_mask_bool = repeat(torch.arange(seqlen, device='cuda'), 's -> b s', b=batch_size) > lengths
# import pdb; pdb.set_trace()
attention_mask = torch.zeros(batch_size, seqlen, device='cuda', dtype=dtype)
attention_mask = rearrange(attention_mask, 'b s -> b 1 1 s')

cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                            device=x.device)

qkv_flash = rearrange(Wqkv(x), 'b s (t h d) -> (b s) t h d', t=3, h=nheads).detach().requires_grad_()

print("@ qkv unpad shape: ",qkv_flash.shape)
qkv = rearrange(Wqkv(x), 'b s (t h d) -> b s t h d', t=3, h=nheads).detach().requires_grad_()
print("@ qkv shape: ",qkv.shape)
fn = lambda qkv_flash: flash_attn_unpadded_qkvpacked_func(
    qkv_flash, cu_seqlens, seqlen, dropout_p, causal=causal
)
benchmark_forward(fn, qkv_flash, repeats=repeats, desc='FlashAttention')
fn = lambda qkv: attention_ref(qkv, dropout_p, causal=causal)
benchmark_forward(fn, qkv, repeats=repeats, desc='PyTorch Standard Attention')

# == error test 
fn_flash = lambda qkv_flash: flash_attn_unpadded_qkvpacked_func(
    qkv_flash, cu_seqlens, seqlen, dropout_p, causal=causal
)
fn_reference = lambda qkv: attention_ref(qkv, dropout_p, causal=causal)

flash_result=fn_flash(qkv_flash).reshape(batch_size,seqlen,nheads,d)
reference_result=fn_reference(qkv)
# import pdb; pdb.set_trace()
# print(flash_result-reference_result)
print(torch.max(torch.abs(flash_result - reference_result).flatten()))
