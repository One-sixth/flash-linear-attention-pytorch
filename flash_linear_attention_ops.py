'''
最开始的内存节省线性注意力实现
可以灵活支持变长输入
一般情况下，建议使用这个算子，因为不需要填充长度到指定的倍数。
'''

import torch
from normal_linear_attention_ops import normal_linear_attention, normal_linear_attention_no_mask


BLOCK_SIZE = 128


def linear_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, M: torch.Tensor):
    A = Q @ K.transpose(-1, -2) * M
    O = A @ V
    return O


def linear_attention_backward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, M: torch.Tensor, dO: torch.Tensor):
    A = Q @ K.transpose(-1, -2)

    dAm = dO @ V.transpose(-1, -2)
    dM = dAm * A
    dA = dAm * M

    Am = A * M
    dV = Am.transpose(-1, -2) @ dO
    dK = dA.transpose(-1, -2) @ Q
    dQ = dA @ K
    return dQ, dK, dV, dM


@torch.no_grad()
def flash_linear_attention_forward(Q, K, V, M):
    O = torch.zeros_like(Q)

    Q_BLOCK_SIZE = BLOCK_SIZE
    KV_BLOCK_SIZE = BLOCK_SIZE

    Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=-2)
    K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=-2)
    V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=-2)
    M_BLOCKS = tuple(torch.split(mc, KV_BLOCK_SIZE, dim=-1) for mc in torch.split(M, Q_BLOCK_SIZE, dim=-2))

    Tr = len(Q_BLOCKS)
    Tc = len(K_BLOCKS)

    O_BLOCKS = torch.split(O, Q_BLOCK_SIZE, dim=-2)

    for i in range(Tr):
        Qi = Q_BLOCKS[i]
        Oi = O_BLOCKS[i]

        for j in range(Tc):
            Kj = K_BLOCKS[j]
            Vj = V_BLOCKS[j]
            Mij = M_BLOCKS[i][j]

            Oi += linear_attention_forward(Qi, Kj, Vj, Mij)

    return O


@torch.no_grad()
def flash_linear_attention_backward(Q, K, V, M, dO):
    # 处理 MQA 的情况
    is_single_K = Q.shape[1] != K.shape[1] and K.shape[1] == 1
    is_single_V = Q.shape[1] != V.shape[1] and V.shape[1] == 1
    is_single_M_B = Q.shape[0] != M.shape[0] and M.shape[0] == 1
    is_single_M_H = Q.shape[1] != M.shape[1] and M.shape[1] == 1
    #
    Q_BLOCK_SIZE = BLOCK_SIZE
    KV_BLOCK_SIZE = BLOCK_SIZE

    Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=-2)
    K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=-2)
    V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=-2)
    M_BLOCKS = tuple(torch.split(mc, KV_BLOCK_SIZE, dim=-1) for mc in torch.split(M, Q_BLOCK_SIZE, dim=-2))

    Tr = len(Q_BLOCKS)
    Tc = len(K_BLOCKS)

    dO_BLOCKS = torch.split(dO, Q_BLOCK_SIZE, dim=-2)

    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    dM = torch.zeros_like(M)

    dQ_BLOCKS = torch.split(dQ, Q_BLOCK_SIZE, dim=-2)
    dK_BLOCKS = torch.split(dK, KV_BLOCK_SIZE, dim=-2)
    dV_BLOCKS = torch.split(dV, KV_BLOCK_SIZE, dim=-2)
    dM_BLOCKS = tuple(torch.split(mc, KV_BLOCK_SIZE, dim=-1) for mc in torch.split(dM, Q_BLOCK_SIZE, dim=-2))

    for j in range(Tc):
        Kj = K_BLOCKS[j]
        Vj = V_BLOCKS[j]

        dKj = dK_BLOCKS[j]
        dVj = dV_BLOCKS[j]

        for i in range(Tr):
            Qi = Q_BLOCKS[i]
            dOi = dO_BLOCKS[i]
            Mij = M_BLOCKS[i][j]
            dQi = dQ_BLOCKS[i]
            dMij = dM_BLOCKS[i][j]

            _dQi, _dKj, _dVj, _dMij = linear_attention_backward(Qi, Kj, Vj, Mij, dOi)

            if is_single_K:
                _dKj = _dKj.sum(1, keepdim=True)
            if is_single_V:
                _dVj = _dVj.sum(1, keepdim=True)

            if is_single_M_B and is_single_M_H:
                torch.sum(_dMij, [0, 1], True, out=_dMij[0:1, 0:1])
                _dMij = _dMij[0:1, 0:1]

            elif is_single_M_B:
                torch.sum(_dMij, 0, True, out=_dMij[0:1])
                _dMij = _dMij[0:1]

            elif is_single_M_H:
                torch.sum(_dMij, 1, True, out=_dMij[:, 0:1])
                _dMij = _dMij[:, 0:1]

            dQi += _dQi
            dKj += _dKj
            dVj += _dVj
            dMij += _dMij

    return dQ, dK, dV, dM


class _FlashLinearAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, M):
        O = flash_linear_attention_forward(Q, K, V, M)
        ctx.save_for_backward(Q, K, V, M)
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, M = ctx.saved_tensors
        dQ, dK, dV, dM = flash_linear_attention_backward(Q, K, V, M, dO)
        return dQ, dK, dV, dM


def flash_linear_attention(Q, K, V, M):
    assert Q.ndim == K.ndim == V.ndim == 4
    assert M is None or M.ndim == 4
    assert Q.shape[0] == K.shape[0] == V.shape[0]
    assert Q.shape[-1] == K.shape[-1]
    assert K.shape[-2] == V.shape[-2]

    if M is None:
        # 如果没有mask，可以直接使用矩阵乘法，最快和最节省
        out = normal_linear_attention_no_mask(Q, K, V)
    else:
        out = _FlashLinearAttentionFunc.apply(Q, K, V, M)
    return out


def test_flash_linear_attention():
    import time

    device = 'cuda'
    # device = 'cpu'
    dtype = torch.float32
    # dtype = torch.float64

    batch_size = 16
    seq_len = 1024
    dim = 32
    n_head = 8

    Q = torch.randn(batch_size, n_head, seq_len, dim, requires_grad=True, dtype=dtype, device=device)
    # K = torch.randn(batch_size, n_head, seq_len, dim, requires_grad=True, dtype=dtype, device=device)
    K = torch.randn(batch_size, 1, seq_len, dim, requires_grad=True, dtype=dtype, device=device)
    V = torch.randn(batch_size, n_head, seq_len, dim, requires_grad=True, dtype=dtype, device=device)
    # M = torch.randint(0, 2, (1, 1, seq_len, seq_len), device=device, dtype=dtype)
    # M = torch.randn(batch_size, n_head, seq_len, seq_len, requires_grad=True, device=device, dtype=dtype)
    M = torch.randn(batch_size, 1, seq_len, seq_len, requires_grad=True, device=device, dtype=dtype)

    # ---------------------------------------------------------------------------------------------------------------

    def check_close(A, B):
        b = torch.allclose(A, B, rtol=1e-2, atol=1e-2)
        print(b, (A - B).abs().max().item())
        return b

    # 测试正确性
    O_normal = normal_linear_attention(Q, K, V, M)
    dQ_normal, dK_normal, dV_normal, dM_normal = torch.autograd.grad(O_normal, inputs=[Q, K, V, M], grad_outputs=torch.ones_like(O_normal))
    #
    O_flash = flash_linear_attention_forward(Q, K, V, M)
    #
    dQ_flash, dK_flash, dV_flash, dM_flash = flash_linear_attention_backward(Q, K, V, M, torch.ones_like(O_normal))
    # 检测 输出和梯度是否正确
    check_close(O_normal, O_flash)
    check_close(dQ_normal, dQ_flash)
    check_close(dK_normal, dK_flash)
    check_close(dV_normal, dV_flash)
    check_close(dM_normal, dM_flash)

    # 检测算子
    O_op = flash_linear_attention(Q, K, V, M)
    dQ_op, dK_op, dV_op, dM_op = torch.autograd.grad(O_op, inputs=[Q, K, V, M], grad_outputs=torch.ones_like(O_op))
    # 检测算子是否正确
    check_close(O_normal, O_op)
    check_close(dQ_normal, dQ_op)
    check_close(dK_normal, dK_op)
    check_close(dV_normal, dV_op)
    check_close(dM_normal, dM_op)
    # 检测结束

    # ---------------------------------------------------------------------------------------------------------------

    # 测试性能
    n = 100

    ##
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start1 = time.perf_counter()
    for _ in range(n):
        out1 = flash_linear_attention(Q, K, V, M)
        out1.sum().backward()

    torch.cuda.synchronize()
    end1 = time.perf_counter()
    print('flash', 'time', end1 - start1)

    ##
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start2 = time.perf_counter()
    for _ in range(n):
        out1 = normal_linear_attention(Q, K, V, M)
        out1.sum().backward()

    torch.cuda.synchronize()
    end2 = time.perf_counter()
    print('normal', 'time', end2 - start2)

    return True


if __name__ == "__main__":
    test_flash_linear_attention()
