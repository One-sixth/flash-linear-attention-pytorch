'''
参考自 op_3 的实现
改为cuda算子，受限很多，并不如 ops_3 通用
例如 C_SIZE 不能太大，超过64就会因为静态内存不足而编译失败

节省显存最多，但速度约为正常算子的 1/3
目前只支持 float32 数据类型。
float16 的支持受设备sm版本限制
float64 的支持则会吃掉太多的静态内存，导致 block size 只能为16或更小

额外，可以参考该cuda算子的实现，改变为其他线性注意力实现，例如 relu2
'''

import math
import torch
import torch.nn.functional as F
from la_kernel.get_la_op import get_cuda_op
from normal_linear_attention_ops import normal_linear_attention, normal_linear_attention_no_mask


# 算子缓存，因为算子编译需要固定值的 Q_BLOCK_SIZE， KV_BLOCK_SIZE 和 C_SIZE
# (Q_BLOCK_SIZE， KV_BLOCK_SIZE 和 C_SIZE): op
_ops_dict = {}


@torch.no_grad()
def flash_linear_attention_forward(Q, K, V, M, *, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    O = torch.zeros_like(Q)

    B, qH, qL, C = Q.shape
    kH, kL = K.shape[1:3]
    vH, vL = V.shape[1:3]
    mB, mH = M.shape[:2]

    op_name = (Q_BLOCK_SIZE, KV_BLOCK_SIZE, C)
    if op_name not in _ops_dict:
        _ops_dict[op_name] = get_cuda_op(Q_BLOCK_SIZE, KV_BLOCK_SIZE, C, Q.dtype)
    la_op = _ops_dict[op_name]
    
    if kH != qH:
        K = K.expand(-1, qH, -1, -1)
    if vH != qH:
        V = V.expand(-1, qH, -1, -1)
    if mB != B:
        M = M.expand(B, -1, -1, -1)
    if mH != qH:
        M = M.expand(-1, qH, -1, -1)

    la_op.forward(Q, K, V, M, O)
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

    #
    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    dM = torch.zeros_like(M)
    #

    op_name = (Q_BLOCK_SIZE, KV_BLOCK_SIZE, C)
    if op_name not in _ops_dict:
        _ops_dict[op_name] = get_cuda_op(Q_BLOCK_SIZE, KV_BLOCK_SIZE, C, Q.dtype)
    la_op = _ops_dict[op_name]

    # 因为我在 cuda算子里使用原子加法，所以这里可以直接用expand，不用担心写入冲突
    if kH != qH:
        K = K.expand(-1, qH, -1, -1)
        dK = dK.expand(-1, qH, -1, -1)
    if vH != qH:
        V = V.expand(-1, qH, -1, -1)
        dV = dV.expand(-1, qH, -1, -1)
    if mB != B:
        M = M.expand(B, -1, -1, -1)
        dM = dM.expand(B, -1, -1, -1)
    if mH != qH:
        M = M.expand(-1, qH, -1, -1)
        dM = dM.expand(-1, qH, -1, -1)

    la_op.backward(Q, K, V, M, dO, dQ, dK, dV, dM)

    if is_single_K:
        dK = dK[:, :1, :, :]

    if is_single_V:
        dV = dV[:, :1, :, :]

    if is_single_M_B:
        dM = dM[:1, :, :, :]
    if is_single_M_H:
        dM = dM[:, :1, :, :]

    return dQ, dK, dV, dM


def calc_best_block_size(L: int):
    # cuda 算子的最佳大小只有32
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
    # dtype = torch.bfloat16

    batch_size = 16
    seq_len = 1024
    dim = 32
    n_head = 8
    # dim = 768
    # n_head = 1

    Q = torch.randn(batch_size, n_head, seq_len, dim, requires_grad=True, dtype=dtype, device=device)
    K = torch.randn(batch_size, n_head, seq_len, dim, requires_grad=True, dtype=dtype, device=device)
    # K = torch.randn(batch_size, 1, seq_len, dim, requires_grad=True, dtype=dtype, device=device)
    V = torch.randn(batch_size, n_head, seq_len, dim, requires_grad=True, dtype=dtype, device=device)
    # M = torch.randint(0, 2, (batch_size, n_head, seq_len, seq_len), requires_grad=True, device=device, dtype=dtype)
    M = torch.randn(batch_size, n_head, seq_len, seq_len, requires_grad=True, device=device, dtype=dtype)
    # M = torch.randn(1, 1, seq_len, seq_len, requires_grad=True, device=device, dtype=dtype)

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
