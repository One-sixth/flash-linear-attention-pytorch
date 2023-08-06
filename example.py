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
