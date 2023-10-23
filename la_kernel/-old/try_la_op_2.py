import os
import sys
import time
import torch


if sys.platform == 'win32':
    os.environ["PATH"] = r'Z:\Software\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.37.32822\bin\Hostx64\x64;' + os.environ["PATH"]


_kernel_root = os.path.dirname(__file__)

la_cuda = torch.utils.cpp_extension.load(
    name=f"linear_attention_op",
    sources=[
        _kernel_root + "/linear_attention_define.cpp",
        _kernel_root + "/linear_attention_kernel.cu"
    ],
    verbose=True,
    extra_cuda_cflags=[
        # "-res-usage",
        "--use_fast_math",
        "-O3",
        "--extra-device-vectorization",
        # f"-DTmax={T_MAX}",
        # "--maxrregcount 60",
        # "-Xptxas -dlcm=ca",
    ]
)

# single

# q = torch.rand(10, 32, 50) * 10
# k = torch.rand(10, 32, 50) * 10
# v = torch.rand(10, 32, 50)
# m = torch.rand(10, 32, 32)
# o = torch.zeros(10, 32, 50)

# q,k,v,m,o = [a.cuda() for a in [q,k,v,m,o]]

# la_cuda.myMatmul(q,k,v,m,o)
# o2 = (q @ k.swapdims(-1,-2) * m) @ v

# print(o.shape)
# print(o2.shape)

# print(o[0, 0, :10])
# print(o2[0, 0, :10])
# print((o-o2).abs().max())


# multi


# q = torch.rand(1, 1, 32*12, 50)
# k = torch.rand(1, 1, 32*12, 50)
# v = torch.rand(1, 1, 32*12, 50)
# m = torch.rand(1, 1, 32*12, 32*12)
# o = torch.zeros(1, 1, 32*12, 50)

# q,k,v,m,o = [a.cuda() for a in [q,k,v,m,o]]

# la_cuda.forward(q,k,v,m,o)
# o2 = (q @ k.swapdims(-1,-2) * m) @ v

# print(o.shape)
# print(o2.shape)
# print(o[0, 0, 0, :10])
# print(o2[0, 0, 0, :10])
# print((o-o2).abs().max())

# single bw


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

# q = torch.rand(10, 32, 50) * 10
# k = torch.rand(10, 32, 50) * 10
# v = torch.rand(10, 32, 50)
# m = torch.rand(10, 32, 32)
# do = torch.ones(10, 32, 50)
# #
# dq = torch.zeros(10, 32, 50)
# dk = torch.zeros(10, 32, 50)
# dv = torch.zeros(10, 32, 50)
# dm = torch.zeros(10, 32, 32)

# q,k,v,m,do,dq,dk,dv,dm = [a.cuda() for a in [q,k,v,m,do,dq,dk,dv,dm]]

# la_cuda.cuda_bw(q,k,v,m,do, dq, dk, dv, dm)
# dq2, dk2, dv2, dm2 = linear_attention_backward(q, k, v, m, do)

# # print(o.shape)
# # print(o2.shape)

# print(dq[0, 0, :10])
# print(dq2[0, 0, :10])
# print((dq-dq2).abs().max())
# print((dk-dk2).abs().max())
# print((dv-dv2).abs().max())
# print((dm-dm2).abs().max())


# multi bw
# 注意，需要为 L 长度需要为 32 的倍数，因为 32 是 cuda 内定义的分块大小
B = 12
H = 8
qL = 32*32*1
kL = 32*32*1
C = 256


q = torch.rand(B, H, qL, C)
k = torch.rand(B, H, kL, C)
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
torch.cuda.synchronize();
t1 = time.time()
la_cuda.forward(q,k,v,m,o)
torch.cuda.synchronize();
t2 = time.time()
la_cuda.backward(q,k,v,m,do,dq,dk,dv,dm)
torch.cuda.synchronize();
t3 = time.time()
o2 = linear_attention_forward(q, k, v, m)
torch.cuda.synchronize();
t4 = time.time()
dq2, dk2, dv2, dm2 = linear_attention_backward(q, k, v, m, do)
torch.cuda.synchronize();
t5 = time.time()

print(t2-t1, t3-t2, t4-t3, t5-t4)

# # print(o.shape)
# # print(o2.shape)

print(o[0, 0, 0, :10])
print(o2[0, 0, 0, :10])
print(dq[0, 0, 0, :10])
print(dq2[0, 0, 0, :10])

print((o-o2).abs().max())
print((dq-dq2).abs().max())
print((dk-dk2).abs().max())
print((dv-dv2).abs().max())
print((dm-dm2).abs().max())

torch.cuda.synchronize();
print(1)