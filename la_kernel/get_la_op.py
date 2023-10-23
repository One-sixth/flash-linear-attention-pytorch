import os
import sys
import time

if sys.platform == 'win32':
    os.environ["PATH"] = r'Z:\Software\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64;' + os.environ["PATH"]

import torch
from torch.utils.cpp_extension import load


_kernel_root = os.path.dirname(__file__)


def get_cuda_op(Q_BLOCK_SIZE, KV_BLOCK_SIZE, C_SIZE, dtype):
    # 阻止非 float32 编译，因为无法编译
    if dtype != torch.float32:
        raise AssertionError('Only support dtype float32.')

    if dtype == torch.float32:
        DTYPE = 'float'
    elif dtype == torch.float16:
        DTYPE = 'half'
    elif dtype == torch.float64:
        DTYPE = 'double'
    elif dtype == torch.bfloat16:
        DTYPE = '__nv_bfloat16'
    else:
        raise AssertionError('Unknow DTYPE')

    la_cuda = load(
        name=f"la_C{C_SIZE}_Q{Q_BLOCK_SIZE}_KV{KV_BLOCK_SIZE}_D{DTYPE}",
        sources=[
            _kernel_root + "/linear_attention_define.cpp",
            _kernel_root + "/linear_attention_kernel.cu"
        ],
        verbose=True,
        # extra_cflags=[
        #     "-fopenmp",
        # ],
        extra_cuda_cflags=[
            # "-res-usage",
            "--use_fast_math",
            "-O3",
            "--extra-device-vectorization",
            "-w",
            f"-D_C_SIZE={C_SIZE},_Q_BLOCK_SIZE={Q_BLOCK_SIZE},_KV_BLOCK_SIZE={KV_BLOCK_SIZE},_DTYPE={DTYPE}",
            # "--maxrregcount 60",
            # "-Xptxas -dlcm=ca",
        ]
    )

    return la_cuda


if __name__ == '__main__':
    # ---------------------------------------------------------------------
    C_SIZE = 32
    Q_BLOCK_SIZE = 32
    KV_BLOCK_SIZE = 32

    la_cuda = get_cuda_op(Q_BLOCK_SIZE, KV_BLOCK_SIZE, C_SIZE, torch.float32)

    def linear_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, M: torch.Tensor):
        A = Q @ K.transpose(-1, -2)
        Am = A * M
        O = Am @ V
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

    # ---------------------------------------------------------------------

    # multi
    # 注意，需要为 L 长度需要为 32 的倍数，因为 32 是 cuda 内定义的分块大小
    B = 16
    H = 8
    qL = Q_BLOCK_SIZE * 16
    kL = KV_BLOCK_SIZE * 16
    C = C_SIZE


    q = torch.rand(B, H, qL, C) * 2
    k = torch.rand(B, H, kL, C) * 2
    v = torch.rand(B, H, kL, C)
    m = torch.rand(B, H, qL, kL)
    o = torch.zeros(B, H, qL, C)
    do = torch.ones(B, H, qL, C)
    #
    dq = torch.zeros(B, H, qL, C)
    dk = torch.zeros(B, H, kL, C)
    dv = torch.zeros(B, H, kL, C)
    dm = torch.zeros(B, H, qL, kL)

    q,k,v,m,o,do,dq,dk,dv,dm = [a.cuda() for a in [q,k,v,m,o,do,dq,dk,dv,dm]]

    print('start')
    torch.cuda.synchronize()
    t1 = time.time()
    la_cuda.forward(q,k,v,m,o)
    torch.cuda.synchronize()
    t2 = time.time()
    la_cuda.backward(q,k,v,m,do,dq,dk,dv,dm)
    torch.cuda.synchronize()
    t3 = time.time()
    o2 = linear_attention_forward(q, k, v, m)
    torch.cuda.synchronize()
    t4 = time.time()
    dq2, dk2, dv2, dm2 = linear_attention_backward(q, k, v, m, do)
    torch.cuda.synchronize()
    t5 = time.time()

    print(t2-t1, t3-t2, t4-t3, t5-t4)

    # # print(o.shape)
    # # print(o2.shape)

    print(o[0, 0, 0, :10])
    print(o2[0, 0, 0, :10])
    print(dq[0, 0, 0, :10])
    print(dq2[0, 0, 0, :10])
    print(dk[0, 0, 0, :10])
    print(dk2[0, 0, 0, :10])

    print((o-o2).abs().max())
    print((dq-dq2).abs().max())
    print((dk-dk2).abs().max())
    print((dv-dv2).abs().max())
    print((dm-dm2).abs().max())

    torch.cuda.synchronize();
    print(1)
