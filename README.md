# flash-linear-attention-pytorch
Pure Python implementation of flash linear attention operators in TransnormerLLM.  
For learning purposes.  
~~If you want to use it for training the model, you may need to modify it to a CUDA or Triton implementation, otherwise it will be slow.~~  
Even though I wrote about implementing the CUDA operator, it is still relatively slow and may require advanced CUDA optimization.  

纯 Pytorch 实现 TransnormerLLM 中快速线性注意力算子。  
用于学习目的。  
~~如果你希望用于训练模型，你可能要修改为 CUDA 或 Triton 的实现，不然会很慢。~~  
即便我改为CUDA算子实现，仍然比较慢，可能需要高级CUDA优化  

# Update / 更新
2023-10-19  
Add support for MQA  
Support Mask Gradient Calculation  
Updated to 5 writing methods  
增加支持MQA  
支持 Mask 梯度计算  
更新为5种写法  


# Introduction to 5 types of operators / 5种写法的算子介绍
normal_linear_attention_ops.py  
原始方式，显存占用最大，速度最快  
可读性：最佳  
显存消耗：1X (O^2)  
速度：1X  


flash_linear_attention_ops.py  
原始分块方式  
内部使用torch.split，不需要填充到指定长度  
可读性：佳  
显存消耗：0.7X  
速度：0.5X  


flash_linear_attention_ops_2.py  
基于 flash_linear_attention_ops.py 改为块索引方式，略快一丁点  
内部需要填充到指定倍数长度  
可读性：佳  
显存消耗：0.7X  
速度：0.505X  


flash_linear_attention_ops_3.py  
基于 flash_linear_attention_ops_2.py 加入显式内存复用方式，略快一丁点  
即在一开始就分配所有需要的显存，在计算过程中，完全不需要新的显存分配  
内部需要填充到指定倍数长度  
可读性：中  
显存消耗：0.7X  
速度：0.51X  


flash_linear_attention_ops_4.py  
基于 flash_linear_attention_ops_3.py，改为CUDA/C++算子方式  
本人的CUDA/C++技术有限，没有精力继续研究了  
内部需要填充到指定倍数长度  
限制很多，不支持float32以外的数据类型  
可以作为其他类型线性注意力的参考实现  
算子已经通过 pytorch2.1 + CUDA12.1 环境测试  
可读性：较差  
显存消耗：0.3X  
速度：0.33X  


# Note / 附注
2023-10-19  

For linear attention, it seems better to directly increase the number of attention channels rather than increasing the number of attention heads.  
For example, 12 heads with 64 dimensions is not as good as 1 head with 768 dimensions  
对于线性注意力，相比增加注意力头的数量，直接增加注意力通道数量似乎更佳  
例如 12头64维度 不如 1头768维度 的性能好  

----
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
