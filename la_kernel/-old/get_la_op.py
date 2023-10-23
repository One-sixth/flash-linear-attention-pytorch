import os
import sys

if sys.platform == 'win32':
    os.environ["PATH"] = r'Z:\Software\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64;' + os.environ["PATH"]

import torch
from torch.utils.cpp_extension import load


_kernel_root = os.path.dirname(__file__)


def get_cuda_op(T_MAX=512):
    '''
    :return:
    '''

    la_cuda = load(name=f"la_{T_MAX}",
                    sources=[
                        _kernel_root + "/la_op.cpp",
                        _kernel_root + "/la_cuda.cu"
                    ],
                    verbose=True,
                    extra_cuda_cflags=[
                        "-res-usage",
                        "--use_fast_math",
                        "-O3",
                        "--extra-device-vectorization",
                        f"-DTmax={T_MAX}",
                        # "--maxrregcount 60",
                        # "-Xptxas -O3",
                    ])

    class LA(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, w, u, k, v):
            ctx.B = B
            ctx.T = T
            ctx.C = C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            #
            w = -torch.exp(w.contiguous())
            u = u.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            #
            y = torch.empty((B, T, C), device=w.device, memory_format=torch.contiguous_format)
            la_cuda.forward(B, T, C, w, u, k, v, y)
            ctx.save_for_backward(w, u, k, v, y)
            return y

        @staticmethod
        def backward(ctx, gy):
            B = ctx.B
            T = ctx.T
            C = ctx.C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            w, u, k, v, y = ctx.saved_tensors
            gw = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)
            gu = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)
            gk = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)
            gv = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)
            la_cuda.backward(B, T, C, w, u, k, v, y, gy.contiguous(), gw, gu, gk, gv)
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)

    return LA


def la_cpu(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, M: torch.Tensor):
    A = Q @ K.transpose(-1, -2) * M
    O = A @ V
    return O


_LA_CUDA = None


def wkv_apply(B, T, C, w, u, k, v):
    global _LA_CUDA

    if w.device.type == 'cuda':
        if _LA_CUDA is None:
            _LA_CUDA = get_cuda_op()
        return _LA_CUDA.apply(B, T, C, w, u, k, v)

    else:
        return la_cpu(w, u, k, v)[0]


if _LA_CUDA is None:
    _LA_CUDA = get_cuda_op()
