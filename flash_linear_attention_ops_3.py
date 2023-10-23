'''
基于op_2，改为重用已有内存区域，可以轻微加速和减少初次内存消耗
BLOCK_SIZE 改为动态计算，也可以加速
消耗显存大约为原来的0.7倍，长度越长减少消耗越明显。速度大约为原来的0.5倍

同时，这个实现也是 op_4 的cuda实现的参考
'''

import math
import torch
import torch.nn.functional as F
from normal_linear_attention_ops import normal_linear_attention, normal_linear_attention_no_mask


def linear_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, M: torch.Tensor,
                             A: torch.Tensor, O: torch.Tensor):
    # A = Q @ K.transpose(-1, -2)
    torch.matmul(Q, K.transpose(-1, -2), out=A)
    # Am = A * M
    Am = torch.mul(A, M, out=A)
    # O = Am @ V
    torch.matmul(Am, V, out=O)


def linear_attention_backward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, M: torch.Tensor, dO: torch.Tensor,
                              A: torch.Tensor, dAm: torch.Tensor, dA: torch.Tensor,
                              dQ: torch.Tensor, dK: torch.Tensor, dV: torch.Tensor, dM: torch.Tensor):
    # A = Q @ K.transpose(-1, -2)
    torch.matmul(Q, K.transpose(-1, -2), out=A)

    # dAm = dO @ V.transpose(-1, -2)
    torch.matmul(dO, V.transpose(-1, -2), out=dAm)

    # dM = dAm * A
    torch.mul(dAm, A, out=dM)
    # dA = dAm * M
    torch.mul(dAm, M, out=dA)

    # Am = A * M
    Am = torch.mul(A, M, out=A)

    # dV = Am.transpose(-1, -2) @ dO
    torch.matmul(Am.transpose(-1, -2), dO, out=dV)

    # dK = dA.transpose(-1, -2) @ Q
    torch.matmul(dA.transpose(-1, -2), Q, out=dK)

    # dQ = dA @ K
    torch.matmul(dA, K, out=dQ)


@torch.no_grad()
def flash_linear_attention_forward(Q, K, V, M, *, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    O = torch.zeros_like(Q)

    B, qH, qL, C = Q.shape
    kH, kL = K.shape[1:3]
    vH, vL = V.shape[1:3]
    mB, mH = M.shape[:2]
    
    if kH != qH:
        K = K.expand(-1, qH, -1, -1)
    if vH != qH:
        V = V.expand(-1, qH, -1, -1)
    if mB != B:
        M = M.expand(B, -1, -1, -1)
    if mH != qH:
        M = M.expand(-1, qH, -1, -1)

    # Q_BLOCK_SIZE = BLOCK_SIZE
    # KV_BLOCK_SIZE = BLOCK_SIZE

    qN = qL // Q_BLOCK_SIZE
    kN = kL // KV_BLOCK_SIZE

    Q_BLOCKS = Q.reshape(B, qH, qN, Q_BLOCK_SIZE, C)
    K_BLOCKS = K.reshape(B, qH, kN, KV_BLOCK_SIZE, C)
    V_BLOCKS = V.reshape(B, qH, kN, KV_BLOCK_SIZE, C)
    M_BLOCKS = M.reshape(B, qH, qN, Q_BLOCK_SIZE, kN, KV_BLOCK_SIZE).swapdims(3, 4)

    O_BLOCKS = O.reshape(B, qH, qN, Q_BLOCK_SIZE, C)

    # cache
    A_tmp: torch.Tensor = torch.empty(B, qH, Q_BLOCK_SIZE, KV_BLOCK_SIZE, device=Q.device, dtype=Q.dtype)
    O_tmp: torch.Tensor = torch.empty(B, qH, Q_BLOCK_SIZE, C, device=Q.device, dtype=Q.dtype)
    #

    for i in range(qN):
        Qi = Q_BLOCKS[:, :, i]
        Oi = O_BLOCKS[:, :, i]

        for j in range(kN):
            Kj = K_BLOCKS[:, :, j]
            Vj = V_BLOCKS[:, :, j]
            Mij = M_BLOCKS[:, :, i, j]

            linear_attention_forward(Qi, Kj, Vj, Mij, A_tmp, O_tmp)
            Oi += O_tmp

    return O


@torch.no_grad()
def flash_linear_attention_backward(Q, K, V, M, dO, *, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    # 处理 MQA 的情况
    is_single_K = Q.shape[1] != K.shape[1] and K.shape[1] == 1
    is_single_V = Q.shape[1] != V.shape[1] and V.shape[1] == 1
    is_single_M_B = Q.shape[0] != M.shape[0] and M.shape[0] == 1
    is_single_M_H = Q.shape[1] != M.shape[1] and M.shape[1] == 1
    #
    B, qH, qL, C = Q.shape
    kH, kL = K.shape[1:3]
    vH, vL = V.shape[1:3]
    mB, mH = M.shape[:2]

    # Q_BLOCK_SIZE = BLOCK_SIZE
    # KV_BLOCK_SIZE = BLOCK_SIZE

    qN = qL // Q_BLOCK_SIZE
    kN = kL // KV_BLOCK_SIZE

    Q_BLOCKS = Q.reshape(B, qH, qN, Q_BLOCK_SIZE, C)
    K_BLOCKS = K.reshape(B, kH, kN, KV_BLOCK_SIZE, C)
    V_BLOCKS = V.reshape(B, kH, kN, KV_BLOCK_SIZE, C)
    M_BLOCKS = M.reshape(mB, mH, qN, Q_BLOCK_SIZE, kN, KV_BLOCK_SIZE).swapdims(3, 4)

    dO_BLOCKS = dO.reshape(B, qH, qN, Q_BLOCK_SIZE, C)

    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    dM = torch.zeros_like(M)

    dQ_BLOCKS = dQ.reshape(B, qH, qN, Q_BLOCK_SIZE, C)
    dK_BLOCKS = dK.reshape(B, kH, kN, KV_BLOCK_SIZE, C)
    dV_BLOCKS = dV.reshape(B, kH, kN, KV_BLOCK_SIZE, C)
    dM_BLOCKS = dM.reshape(mB, mH, qN, Q_BLOCK_SIZE, kN, KV_BLOCK_SIZE).swapdims(3, 4)

    # cache
    A_tmp: torch.Tensor = torch.empty(B, qH, Q_BLOCK_SIZE, KV_BLOCK_SIZE, device=Q.device, dtype=Q.dtype)
    dAm_tmp: torch.Tensor = torch.empty(B, qH, Q_BLOCK_SIZE, KV_BLOCK_SIZE, device=Q.device, dtype=Q.dtype)
    dA_tmp: torch.Tensor = torch.empty(B, qH, Q_BLOCK_SIZE, KV_BLOCK_SIZE, device=Q.device, dtype=Q.dtype)
    dQ_tmp: torch.Tensor = torch.empty(B, qH, Q_BLOCK_SIZE, C, device=Q.device, dtype=Q.dtype)
    dK_tmp: torch.Tensor = torch.empty(B, qH, KV_BLOCK_SIZE, C, device=Q.device, dtype=Q.dtype)
    dV_tmp: torch.Tensor = torch.empty(B, qH, KV_BLOCK_SIZE, C, device=Q.device, dtype=Q.dtype)
    dM_tmp: torch.Tensor = torch.empty(B, qH, Q_BLOCK_SIZE, KV_BLOCK_SIZE, device=Q.device, dtype=Q.dtype)
    #

    for j in range(kN):
        Kj = K_BLOCKS[:, :, j]
        Vj = V_BLOCKS[:, :, j]

        dKj = dK_BLOCKS[:, :, j]
        dVj = dV_BLOCKS[:, :, j]

        for i in range(qN):
            Qi = Q_BLOCKS[:, :, i]
            dOi = dO_BLOCKS[:, :, i]
            Mij = M_BLOCKS[:, :, i, j]
            dQi = dQ_BLOCKS[:, :, i]
            dMij = dM_BLOCKS[:, :, i, j]

            linear_attention_backward(Qi, Kj, Vj, Mij, dOi, A_tmp, dAm_tmp, dA_tmp, dQ_tmp, dK_tmp, dV_tmp, dM_tmp)

            _dK_bk = dK_tmp
            if is_single_K:
                torch.sum(_dK_bk, 1, True, out=_dK_bk[:, 0:1])
                _dK_bk = _dK_bk[:, 0:1]

            _dV_bk = dV_tmp
            if is_single_V:
                torch.sum(_dV_bk, 1, True, out=_dV_bk[:, 0:1])
                _dV_bk = _dV_bk[:, 0:1]
            
            _dM_tmp = dM_tmp
            if is_single_M_B and is_single_M_H:
                torch.sum(_dM_tmp, [0, 1], True, out=_dM_tmp[0:1, 0:1])
                _dM_tmp = _dM_tmp[0:1, 0:1]

            elif is_single_M_B:
                torch.sum(_dM_tmp, 0, True, out=_dM_tmp[0:1])
                _dM_tmp = _dM_tmp[0:1]

            elif is_single_M_H:
                torch.sum(_dM_tmp, 1, True, out=_dM_tmp[:, 0:1])
                _dM_tmp = _dM_tmp[:, 0:1]

            dQi += dQ_tmp
            dKj += _dK_bk
            dVj += _dV_bk
            dMij += _dM_tmp

    return dQ, dK, dV, dM


def calc_best_block_size(L: int):
    # 最佳大小大约为原来的1/4
    if L > 2048:
        block_size = 512
    elif L > 1024:
        block_size = 256
    elif L > 256:
        block_size = 128
    elif L > 128:
        block_size = 64
    else:
        block_size = 32
    return block_size


class _FlashLinearAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, M, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
        O = flash_linear_attention_forward(Q, K, V, M, Q_BLOCK_SIZE=Q_BLOCK_SIZE, KV_BLOCK_SIZE=KV_BLOCK_SIZE)
        ctx.save_for_backward(Q, K, V, M, torch.as_tensor(Q_BLOCK_SIZE), torch.as_tensor(KV_BLOCK_SIZE))
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, M, Q_BLOCK_SIZE, KV_BLOCK_SIZE = ctx.saved_tensors
        Q_BLOCK_SIZE, KV_BLOCK_SIZE = Q_BLOCK_SIZE.item(), KV_BLOCK_SIZE.item()
        dQ, dK, dV, dM = flash_linear_attention_backward(Q, K, V, M, dO, Q_BLOCK_SIZE=Q_BLOCK_SIZE, KV_BLOCK_SIZE=KV_BLOCK_SIZE)
        return dQ, dK, dV, dM, None, None


def calc_pad_multi(L, fa):
    fa2 = math.ceil(L / fa)
    return int(fa2 * fa - L)


def flash_linear_attention(Q, K, V, M, *, Q_BLOCK_SIZE='auto', KV_BLOCK_SIZE='auto'):
    '''
    :param Q: shape [B,qH,qL,C]
    :param K: shape [B,kH,kL,C]
    :param V: shape [B,kH,kL,C]
    :param M: shape [B,qH,qL,kL]
    :param Q_BLOCK_SIZE:
    :param KV_BLOCK_SIZE:
    :return:
    '''
    assert Q.ndim == K.ndim == V.ndim == 4
    assert M is None or M.ndim == 4
    assert Q.shape[0] == K.shape[0] == V.shape[0]
    assert Q.shape[-1] == K.shape[-1]
    assert K.shape[-2] == V.shape[-2]

    if M is None:
        # 如果没有mask，可以直接使用矩阵乘法，最快和最节省
        out = normal_linear_attention_no_mask(Q, K, V)

    else:
        # 需要对齐，如果不够长，则填充到对齐
        if Q_BLOCK_SIZE == 'auto':
            Q_BLOCK_SIZE = calc_best_block_size(Q.shape[-2])
        if KV_BLOCK_SIZE == 'auto':
            KV_BLOCK_SIZE = calc_best_block_size(K.shape[-2])

        pad_q = calc_pad_multi(Q.shape[-2], Q_BLOCK_SIZE)
        pad_k = calc_pad_multi(K.shape[-2], KV_BLOCK_SIZE)
        pad_v = calc_pad_multi(V.shape[-2], KV_BLOCK_SIZE)
        if pad_q:
            Q = F.pad(Q, [0, 0, 0, pad_q])
        if pad_k:
            K = F.pad(K, [0, 0, 0, pad_k])
        if pad_v:
            V = F.pad(V, [0, 0, 0, pad_v])
        if pad_q or pad_k:
            M = F.pad(M, [0, pad_k, 0, pad_q])

        out = _FlashLinearAttentionFunc.apply(Q, K, V, M, Q_BLOCK_SIZE, KV_BLOCK_SIZE)
        if pad_q:
            out = out[:, :, :-pad_q]

    return out


def test_flash_linear_attention():
    import time

    device = 'cuda'
    # device = 'cpu'
    # dtype = torch.float16
    dtype = torch.float32
    # dtype = torch.float64

    batch_size = 16
    seq_len = 1024
    dim = 32
    n_head = 8
    # dim = 768
    # n_head = 1

    Q = torch.randn(batch_size, n_head, seq_len, dim, requires_grad=True, dtype=dtype, device=device)
    K = torch.randn(batch_size, n_head, seq_len, dim, requires_grad=True, dtype=dtype, device=device)
    V = torch.randn(batch_size, n_head, seq_len, dim, requires_grad=True, dtype=dtype, device=device)
    # M = torch.randint(0, 2, (batch_size, n_head, seq_len, seq_len), requires_grad=True, device=device, dtype=dtype)
    M = torch.randn(batch_size, n_head, seq_len, seq_len, requires_grad=True, device=device, dtype=dtype)

    # ---------------------------------------------------------------------------------------------------------------

    def check_close(A, B):
        b = torch.allclose(A, B, rtol=1e-2, atol=1e-2)
        print(b, (A - B).abs().max().item())
        return b

    # 测试正确性
    O_normal = normal_linear_attention(Q, K, V, M)
    dQ_normal, dK_normal, dV_normal, dM_normal = torch.autograd.grad(O_normal, inputs=[Q, K, V, M], grad_outputs=torch.ones_like(O_normal))

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
