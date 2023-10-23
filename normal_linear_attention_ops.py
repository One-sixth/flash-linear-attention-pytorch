'''
正常情况下的线性注意力
'''

import torch
import torch.nn.functional as F


def normal_linear_attention(Q, K, V, M):
    A = Q @ K.transpose(-1, -2)
    A = A * M
    O = A @ V
    return O


def normal_linear_attention_no_mask(Q, K, V):
    KV = K.transpose(-1, -2) @ V
    O = Q @ KV
    return O
