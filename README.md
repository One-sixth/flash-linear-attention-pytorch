# flash-linear-attention-pytorch
Pure Python implementation of flash linear attention operators in TransnormerLLM.  
For learning purposes.  
If you want to use it for training the model, you may need to modify it to a CUDA or Triton implementation, otherwise it will be slow.  

纯 Pytorch 实现 TransnormerLLM 中快速线性注意力算子。  
用于学习目的。  
如果你希望用于训练模型，你可能要修改为 CUDA 或 Triton 的实现，不然会很慢。  

# Attention / 注意
This operator has accuracy issues and a large error, which is normal.  
This is because the attention matrix has no Activation function, which results in a large value of the attention matrix.  
Special caution is required when using the float16 type.  

这个算子有精度问题，误差较大，是正常的。  
这是因为注意力矩阵没有激活函数，导致注意力矩阵的值很大。  
在使用 float16 类型时需要特别小心。  


This is a simple mitigation method: limit the values of q and k to reduce the possibility of float16 overflow.  
这是一个简单的缓解方法：限制 q 和 k 的值，从而减少float16溢出的可能性。  
```python
q = q / q.norm(-1, keepdim=True)
k = k / k.norm(-1, keepdim=True)
o = linear_attention(q, k, v, m)
```

# Usage / 使用方法
```python
import torch
from flash_linear_attention_ops import flash_linear_attention, normal_linear_attention


batch_size = 16
seq_len = 1024
dim = 64
n_head = 12
device = 'cuda'
dtype = torch.float32


Q = torch.randn(batch_size, n_head, seq_len, dim, requires_grad=True, dtype=dtype, device=device)
K = torch.randn(batch_size, n_head, seq_len, dim, requires_grad=True, dtype=dtype, device=device)
V = torch.randn(batch_size, n_head, seq_len, dim, requires_grad=True, dtype=dtype, device=device)
M = torch.randint(0, 2, (1, 1, seq_len, seq_len), device=device, dtype=dtype)

O_flash = flash_linear_attention(Q, K, V, M)
O_normal = normal_linear_attention(Q, K, V, M)

print('O_flash.shape', O_flash.shape)
print('O_normal.shape', O_normal.shape)

print('O diff', (O_flash - O_normal).abs().max().item())

```

# Reference / 参考引用
https://github.com/OpenNLPLab/TransnormerLLM  
https://github.com/shreyansh26/FlashAttention-PyTorch  
