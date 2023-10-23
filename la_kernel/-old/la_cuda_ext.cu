#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAGraphsUtils.cuh>


#ifndef _C_SIZE
#define _C_SIZE 64
#endif

#ifndef _Q_BLOCK_SIZE
#define _Q_BLOCK_SIZE 32
#endif

#ifndef _KV_BLOCK_SIZE
#define _KV_BLOCK_SIZE 32
#endif

constexpr int Q_BLOCK_SIZE = _Q_BLOCK_SIZE;
constexpr int KV_BLOCK_SIZE = _KV_BLOCK_SIZE;
constexpr int C_SIZE = _C_SIZE;


constexpr int BLOCK_Q_SIZE = Q_BLOCK_SIZE * C_SIZE;
constexpr int BLOCK_KV_SIZE = KV_BLOCK_SIZE * C_SIZE;
constexpr int BLOCK_M_SIZE = Q_BLOCK_SIZE * KV_BLOCK_SIZE;


template <typename integer>
constexpr __host__ __device__ inline integer ceil_div(integer n, integer m) { return (n + m - 1) / m; }


template <
	int m1_row, int m1_col, bool m1_transpose,
	int m2_row, int m2_col, bool m2_transpose,
	int o_row, int o_col,
	typename scalar_t
>
__host__ __device__ inline void my_matmul(
	scalar_t pM1[m1_row][m1_col],
	scalar_t pM2[m2_row][m2_col],
	scalar_t pMO[o_row][o_col]
){
	// 矩阵乘法，m2_transpose 用来处理 Q @ K.T
	// pM1 [L1, C] ; pM2 [L2, C] ; pO [L1, L2]
	constexpr int L1 = m1_transpose ? m1_col : m1_row;
	constexpr int C = m1_transpose ? m1_row : m1_col;
	constexpr int L2 = m2_transpose ? m2_row : m2_col;

	static_assert(L1 == o_row, "L1 must be == o_row");
	static_assert(L2 == o_col, "L2 must be == o_col");

	for (int i=0; i<L1; i++)
		for (int j=0; j<L2; j++)
		{
			scalar_t s = 0;
			for (int ci = 0; ci < C; ++ci)
			{
				auto v1 = m1_transpose ? pM1[ci][i] : pM1[i][ci];
				auto v2 = m2_transpose ? pM2[j][ci] : pM2[ci][j];
				s += v1 * v2;
			}
			pMO[i][j] = s;
		}
}


template <int n, typename scalar_t>
__host__ __device__ inline void mul_arr(
	scalar_t src1[],
	scalar_t src2[],
	scalar_t dst[]
){
	// dst = src1 * src2
	#pragma unroll
	for (int i=0; i<n; i++)
		dst[i] = src1[i] * src2[i];
}


template <int n, typename scalar_t>
__host__ __device__ inline void add_arr(
	scalar_t src1[],
	scalar_t src2[],
	scalar_t dst[]
){
	// dst = src1 + src2
	#pragma unroll
	for (int i=0; i<n; i++)
		dst[i] = src1[i] + src2[i];
}


template <bool dst_to_src, typename scalar_t>
__host__ __device__ inline void add_arr(
	scalar_t src[],
	at::TensorAccessor<scalar_t, 2> dst
){
	// dst += src
	const int row = dst.size(0);
	const int col = dst.size(1);

	// dst = src
	for (int cur_row=0; cur_row<row; cur_row++)
		for (int cur_col=0; cur_col<col; cur_col++)
		{
			if (dst_to_src)
				src[cur_row*col+cur_col] += dst[cur_row][cur_col];
			else
				dst[cur_row][cur_col] += src[cur_row*col+cur_col];
		}
}


template <int n, typename scalar_t>
__host__ __device__ inline void atomic_add_arr(
	scalar_t src[],
	scalar_t dst[]
){
	// dst += src
	#pragma unroll
	for (int i=0; i<n; i++)
		atomicAdd(&dst[i], src[i]);
}


template <bool dst_to_src, typename scalar_t>
__host__ __device__ inline void atomic_add_arr(
	scalar_t src[],
	at::TensorAccessor<scalar_t, 2> dst
){
	// dst += src
	const int row = dst.size(0);
	const int col = dst.size(1);

	// dst = src
	for (int cur_row=0; cur_row<row; cur_row++)
		for (int cur_col=0; cur_col<col; cur_col++)
		{
			if (dst_to_src)
				atomicAdd(&src[cur_row*col+cur_col], dst[cur_row][cur_col]);
			else
				atomicAdd(&dst[cur_row][cur_col], src[cur_row*col+cur_col]);
		}
}


template <int n, typename scalar_t>
__host__ __device__ inline void copy_arr(
	scalar_t src[],
	scalar_t dst[]
){
	// dst = src
	for (int i=0; i<n; i++)
		dst[i] = src[i];
}


template <bool dst_to_src, typename scalar_t>
__host__ __device__ inline void copy_arr(
	scalar_t src[],
	at::TensorAccessor<scalar_t, 2> dst
){
	const int row = dst.size(0);
	const int col = dst.size(1);

	// dst = src
	for (int cur_row=0; cur_row<row; cur_row++)
		for (int cur_col=0; cur_col<col; cur_col++)
		{
			if (dst_to_src)
				src[cur_row*col+cur_col] = dst[cur_row][cur_col];
			else
				dst[cur_row][cur_col] = src[cur_row*col+cur_col];
		}
}


template <int n, typename scalar_t>
__host__ __device__ inline void parallel_copy_arr(
	scalar_t src[],
	scalar_t dst[],
	int n_thread,
	int thread_idx
){
	// A2 = A1
	int tn = ceil_div(n, n_thread);
	int r1 = thread_idx*tn, r2 = (thread_idx+1) * tn;
	#pragma unroll
	for (int i=r1; i<r2 && i<n; i++)
		dst[i] = src[i];
}


template <bool dst_to_src, typename scalar_t>
__host__ __device__ inline void parallel_copy_arr(
	scalar_t src[],
	at::TensorAccessor<scalar_t, 2> dst,
	int n_thread,
	int thread_idx
){
	const int row = dst.size(0);
	const int col = dst.size(1);
	int n = row * col;

	int tn = ceil_div(n, n_thread);
	int r1 = thread_idx*tn, r2 = (thread_idx+1) * tn;

	for (int i=r1; i<r2 && i<n; i++)
	{
		int cur_row = i / col;
		int cur_col = i - (col * cur_row);

		// printf("%d ", i);

		if (dst_to_src)
			src[i] = dst[cur_row][cur_col];
		else
			dst[cur_row][cur_col] = src[i];
	}
}


template <int n, typename scalar_t>
__host__ __device__ inline void parallel_zero_arr(
	scalar_t dst[],
	int n_thread,
	int thread_idx
){
	// dst = 0
	int tn = ceil_div(n, n_thread);
	int r1 = thread_idx*tn, r2 = (thread_idx+1) * tn;
	#pragma unroll
	for (int i=r1; i<r2 && i<n; i++)
		dst[i] = (scalar_t)0;
}


template <typename scalar_t>
__host__ __device__ inline void parallel_zero_arr(
	at::TensorAccessor<scalar_t, 2> dst,
	int n_thread,
	int thread_idx
){
	// dst = 0
	const int row = dst.size(0);
	const int col = dst.size(1);
	int n = row * col;

	int tn = ceil_div(n, n_thread);
	int r1 = thread_idx*tn, r2 = (thread_idx+1) * tn;

	for (int i=r1; i<r2 && i<n; i++)
	{
		int cur_row = i / col;
		int cur_col = i - (col * cur_row);

		dst[cur_row][cur_col] = (scalar_t)0;
	}
}


template <typename scalar_t>
__host__ __device__ inline void linear_attention_forward(
	scalar_t Q[Q_BLOCK_SIZE][C_SIZE],
	scalar_t K[KV_BLOCK_SIZE][C_SIZE],
	scalar_t V[KV_BLOCK_SIZE][C_SIZE],
	scalar_t M[Q_BLOCK_SIZE][KV_BLOCK_SIZE],
	scalar_t O[Q_BLOCK_SIZE][C_SIZE]
){
	// 线性注意力-前向传播函数
	scalar_t A[Q_BLOCK_SIZE][KV_BLOCK_SIZE] = {0};

	// A = Q @ K.transpose(-1, -2)
	my_matmul<
		Q_BLOCK_SIZE, C_SIZE, false,
		KV_BLOCK_SIZE, C_SIZE, true,
		Q_BLOCK_SIZE, KV_BLOCK_SIZE
	>(Q, K, A);

	// A 在后面没用了，这里让Am复用A的空间
	auto &Am = A;
	// Am = A * M
	mul_arr<Q_BLOCK_SIZE*KV_BLOCK_SIZE>((scalar_t*)Am, (scalar_t*)M, (scalar_t*)Am);
	// O = Am @ V
	my_matmul<
		Q_BLOCK_SIZE, KV_BLOCK_SIZE, false,
		KV_BLOCK_SIZE, C_SIZE, false,
		Q_BLOCK_SIZE, C_SIZE
	>(A, V, O);
}


template <typename scalar_t>
__host__ __device__ inline void linear_attention_backward(
	// in
	scalar_t Q[Q_BLOCK_SIZE][C_SIZE],
	scalar_t K[KV_BLOCK_SIZE][C_SIZE],
	scalar_t V[KV_BLOCK_SIZE][C_SIZE],
	scalar_t M[Q_BLOCK_SIZE][KV_BLOCK_SIZE],
	scalar_t dO[Q_BLOCK_SIZE][C_SIZE],
	// out
	scalar_t dQ[Q_BLOCK_SIZE][C_SIZE],
	scalar_t dK[KV_BLOCK_SIZE][C_SIZE],
	scalar_t dV[KV_BLOCK_SIZE][C_SIZE],
	scalar_t dM[Q_BLOCK_SIZE][KV_BLOCK_SIZE]
){
	// 线性注意力-反向传播函数
	scalar_t A[Q_BLOCK_SIZE][KV_BLOCK_SIZE];
	scalar_t dAm[Q_BLOCK_SIZE][KV_BLOCK_SIZE];
	scalar_t dA[Q_BLOCK_SIZE][KV_BLOCK_SIZE];

	// A = Q @ K.transpose(-1, -2)
	my_matmul<
		Q_BLOCK_SIZE, C_SIZE, false,
		KV_BLOCK_SIZE, C_SIZE, true,
		Q_BLOCK_SIZE, KV_BLOCK_SIZE
	>(Q, K, A);

	// dAm = dO @ V.transpose(-1, -2)
	my_matmul<
		Q_BLOCK_SIZE, C_SIZE, false,
		KV_BLOCK_SIZE, C_SIZE, true,
		Q_BLOCK_SIZE, KV_BLOCK_SIZE
	>(dO, V, dAm);
	// dM = dAm * A
	mul_arr<Q_BLOCK_SIZE*KV_BLOCK_SIZE>((scalar_t*)dAm, (scalar_t*)A, (scalar_t*)dM);
	// dA = dAm * M
	mul_arr<Q_BLOCK_SIZE*KV_BLOCK_SIZE>((scalar_t*)dAm, (scalar_t*)M, (scalar_t*)dA);

	// A后面没用了，重用 A ，现在 A 是 Am
	auto& Am = A;
	// Am = A * M
	mul_arr<Q_BLOCK_SIZE*KV_BLOCK_SIZE>((scalar_t*)Am, (scalar_t*)M, (scalar_t*)Am);
	// dV = Am.transpose(-1, -2) @ dO
	my_matmul<
		Q_BLOCK_SIZE, KV_BLOCK_SIZE, true,
		Q_BLOCK_SIZE, C_SIZE, false,
		KV_BLOCK_SIZE, C_SIZE
	>(Am, dO, dV);
	// dK = dA.transpose(-1, -2) @ Q ; 注意这里的dK是子集
	my_matmul<
		Q_BLOCK_SIZE, KV_BLOCK_SIZE, true,
		Q_BLOCK_SIZE, C_SIZE, false,
		KV_BLOCK_SIZE, C_SIZE
	>(dA, Q, dK);
	// dQ = dA @ K ; 注意这里的dQ是子集
	my_matmul<
		Q_BLOCK_SIZE, KV_BLOCK_SIZE, false,
		KV_BLOCK_SIZE, C_SIZE, false,
		Q_BLOCK_SIZE, C_SIZE
	>(dA, K, dQ);
}


template<typename scalar_t>
__global__ void linear_attention_forward_kernel(
	at::PackedTensorAccessor<scalar_t, 5> Q,
	at::PackedTensorAccessor<scalar_t, 5> K,
	at::PackedTensorAccessor<scalar_t, 5> V,
	at::PackedTensorAccessor<scalar_t, 6> M,
	at::PackedTensorAccessor<scalar_t, 5> O,
	int B,
	int H,
	int qL,
	int kL,
	int qN,
	int kN,
	int C
){
	// 获得当前线程分块信息
	int cur_Bi = blockIdx.x / H;
	int cur_Hi = blockIdx.x % H;
	int cur_Qi = blockIdx.y;
	int cur_Ki = threadIdx.x;
	// printf("B%d H%d Q%d K%d\n", cur_Bi, cur_Hi, cur_Qi, cur_Ki);

	// 获得当前线程的分块数据
	at::TensorAccessor<scalar_t, 2> p_block_q = Q[cur_Bi][cur_Hi][cur_Qi];
	at::TensorAccessor<scalar_t, 2> p_block_k = K[cur_Bi][cur_Hi][cur_Ki];
	at::TensorAccessor<scalar_t, 2> p_block_v = V[cur_Bi][cur_Hi][cur_Ki];
	at::TensorAccessor<scalar_t, 2> p_block_m = M[cur_Bi][cur_Hi][cur_Qi][cur_Ki];
	at::TensorAccessor<scalar_t, 2> p_block_o = O[cur_Bi][cur_Hi][cur_Qi];

	// fast data
	// __shared__ scalar_t block_q[Q_BLOCK_SIZE][C_SIZE];
	scalar_t block_q[Q_BLOCK_SIZE][C_SIZE];
	scalar_t block_k[KV_BLOCK_SIZE][C_SIZE];
	scalar_t block_v[KV_BLOCK_SIZE][C_SIZE];
	scalar_t block_m[Q_BLOCK_SIZE][KV_BLOCK_SIZE];
	// 复用 block_q 的空间给 block_o 用
	// scalar_t block_o[Q_BLOCK_SIZE][C_SIZE];
	auto& block_o = block_q;

	// __shared__ scalar_t acc_block_o[Q_BLOCK_SIZE][C_SIZE];
	// parallel_zero_arr<scalar_t, Q_BLOCK_SIZE*C_SIZE>((scalar_t*)acc_block_o, kN, cur_Ki);

	// load data
	copy_arr<true>((scalar_t*)block_q, p_block_q);
	// parallel_copy_arr<true>((scalar_t*)block_q, p_block_q, kN, cur_Ki);
	copy_arr<true>((scalar_t*)block_k, p_block_k);
	copy_arr<true>((scalar_t*)block_v, p_block_v);
	copy_arr<true>((scalar_t*)block_m, p_block_m);

	// 直接复用 block_q 作为 block_o ，节省空间
	linear_attention_forward(block_q, block_k, block_v, block_m, block_o);

	// 呃，直接用原子加法写回全局内存。。。反而比先在SRAM原子加法再复制去全局内存更快。。。
	atomic_add_arr<false>((scalar_t*)block_o, p_block_o);
	//

	// atomic_add_arr<Q_BLOCK_SIZE*C_SIZE>((scalar_t*)block_o, (scalar_t*)acc_block_o);
	// __syncthreads();

	// parallel_copy_arr<false>((scalar_t*)acc_block_o, p_block_o, kN, cur_Ki);
}


template<typename scalar_t>
__global__ void linear_attention_backward_kernel(
	at::PackedTensorAccessor<scalar_t, 5> Q,
	at::PackedTensorAccessor<scalar_t, 5> K,
	at::PackedTensorAccessor<scalar_t, 5> V,
	at::PackedTensorAccessor<scalar_t, 6> M,
	at::PackedTensorAccessor<scalar_t, 5> dO,
	//
	at::PackedTensorAccessor<scalar_t, 5> dQ,
	at::PackedTensorAccessor<scalar_t, 5> dK,
	at::PackedTensorAccessor<scalar_t, 5> dV,
	at::PackedTensorAccessor<scalar_t, 6> dM,
	//
	int B,
	int H,
	int qL,
	int kL,
	int qN,
	int kN,
	int C
){
	// 获得当前线程分块信息
	int cur_Bi = blockIdx.x / H;
	int cur_Hi = blockIdx.x % H;
	// int cur_Qi = blockIdx.y / qN;
	// int cur_Ki = blockIdx.y % qN;
	int cur_Ki = blockIdx.y;
	int cur_Qi = threadIdx.x;
	// printf("B%d H%d Q%d K%d\n", cur_Bi, cur_Hi, cur_Qi, cur_Ki);

	// 获得当前线程的分块数据
	// printf("B%d H%d Q%d K%d\n", cur_Bi, cur_Hi, cur_Qi, cur_Ki);
	at::TensorAccessor<scalar_t, 2> p_block_q = Q[cur_Bi][cur_Hi][cur_Qi];
	at::TensorAccessor<scalar_t, 2> p_block_k = K[cur_Bi][cur_Hi][cur_Ki];
	at::TensorAccessor<scalar_t, 2> p_block_v = V[cur_Bi][cur_Hi][cur_Ki];
	at::TensorAccessor<scalar_t, 2> p_block_m = M[cur_Bi][cur_Hi][cur_Qi][cur_Ki];
	at::TensorAccessor<scalar_t, 2> p_block_do = dO[cur_Bi][cur_Hi][cur_Qi];
	//
	at::TensorAccessor<scalar_t, 2> p_block_dq = dQ[cur_Bi][cur_Hi][cur_Qi];
	at::TensorAccessor<scalar_t, 2> p_block_dk = dK[cur_Bi][cur_Hi][cur_Ki];
	at::TensorAccessor<scalar_t, 2> p_block_dv = dV[cur_Bi][cur_Hi][cur_Ki];
	at::TensorAccessor<scalar_t, 2> p_block_dm = dM[cur_Bi][cur_Hi][cur_Qi][cur_Ki];

	// fast data
	scalar_t block_q[Q_BLOCK_SIZE][C_SIZE];
	__shared__ scalar_t block_k[KV_BLOCK_SIZE][C_SIZE];
	__shared__ scalar_t block_v[KV_BLOCK_SIZE][C_SIZE];
	scalar_t block_m[Q_BLOCK_SIZE][KV_BLOCK_SIZE];
	scalar_t block_do[Q_BLOCK_SIZE][C_SIZE];

	// 这里也是复用 block_q 作为 block_dq，节省空间，也不会出现干扰
	// scalar_t block_dq[Q_BLOCK_SIZE][C_SIZE];
	auto&    block_dq = block_q;
	scalar_t block_dk[KV_BLOCK_SIZE][C_SIZE];
	scalar_t block_dv[KV_BLOCK_SIZE][C_SIZE];
	scalar_t block_dm[Q_BLOCK_SIZE][KV_BLOCK_SIZE];
	//
	// copy_arr<true>((scalar_t*)block_k, p_block_k);
	// copy_arr<true>((scalar_t*)block_v, p_block_v);
	__syncthreads();
	parallel_copy_arr<true>((scalar_t*)block_k, p_block_k, qN, cur_Qi);
	parallel_copy_arr<true>((scalar_t*)block_v, p_block_v, qN, cur_Qi);
	copy_arr<true>((scalar_t*)block_q, p_block_q);
	copy_arr<true>((scalar_t*)block_m, p_block_m);
	copy_arr<true>((scalar_t*)block_do, p_block_do);
	//
	// __shared__ scalar_t acc_block_dq[Q_BLOCK_SIZE][C_SIZE];
	// __shared__ scalar_t acc_block_dk[KV_BLOCK_SIZE][C_SIZE];
	// __shared__ scalar_t acc_block_dv[KV_BLOCK_SIZE][C_SIZE];
	// zero data
	// parallel_zero_arr<scalar_t, Q_BLOCK_SIZE*C_SIZE>((scalar_t*)acc_block_dq, qN, cur_Qi);
	// parallel_zero_arr<scalar_t, KV_BLOCK_SIZE*C_SIZE>((scalar_t*)acc_block_dk, qN, cur_Qi);
	// parallel_zero_arr<scalar_t, KV_BLOCK_SIZE*C_SIZE>((scalar_t*)acc_block_dv, qN, cur_Qi);
	//

	linear_attention_backward(
		block_q, block_k, block_v, block_m, block_do,
		block_dq, block_dk, block_dv, block_dm
	);

	// copy_arr<false>((scalar_t*)block_dm, p_block_dm);

	atomic_add_arr<false>((scalar_t*)block_dq, p_block_dq);
	atomic_add_arr<false>((scalar_t*)block_dk, p_block_dk);
	atomic_add_arr<false>((scalar_t*)block_dv, p_block_dv);
	atomic_add_arr<false>((scalar_t*)block_dm, p_block_dm);
	// 试验无锁的速度
	// add_arr<false>((scalar_t*)block_dq, p_block_dq);
	// add_arr<false>((scalar_t*)block_dk, p_block_dk);
	// add_arr<false>((scalar_t*)block_dv, p_block_dv);
	// add_arr<false>((scalar_t*)block_dm, p_block_dm);

	// __syncthreads();
	// atomic_add_arr<Q_BLOCK_SIZE*C_SIZE>((scalar_t*)block_dq, (scalar_t*)acc_block_dq);
	// atomic_add_arr<KV_BLOCK_SIZE*C_SIZE>((scalar_t*)block_dk, (scalar_t*)acc_block_dk);
	// atomic_add_arr<,KV_BLOCK_SIZE*C_SIZE>((scalar_t*)block_dv, (scalar_t*)acc_block_dv);

	// __syncthreads();
	// parallel_copy_arr<false>((scalar_t*)acc_block_dq, p_block_dq, qN, cur_Qi);
	// parallel_copy_arr<false>((scalar_t*)acc_block_dk, p_block_dk, qN, cur_Qi);
	// parallel_copy_arr<false>((scalar_t*)acc_block_dv, p_block_dv, qN, cur_Qi);

}


template <typename scalar_t>
__global__ void test_main_func(
	at::PackedTensorAccessor<scalar_t, 3> Q,
	at::PackedTensorAccessor<scalar_t, 3> K,
	at::PackedTensorAccessor<scalar_t, 3> V,
	at::PackedTensorAccessor<scalar_t, 3> M,
	at::PackedTensorAccessor<scalar_t, 3> O
) {
	// 矩阵乘法，用来处理 A @ v
	// pM1 [L1, C] ; pM2 [L2, C] ; pO [L1, L2]
	int cur_Bi = blockIdx.x;
	
	// printf("g%d ", cur_Bi);

	at::TensorAccessor<scalar_t, 2> p_block_q = Q[cur_Bi];
	at::TensorAccessor<scalar_t, 2> p_block_k = K[cur_Bi];
	at::TensorAccessor<scalar_t, 2> p_block_v = V[cur_Bi];
	at::TensorAccessor<scalar_t, 2> p_block_m = M[cur_Bi];
	at::TensorAccessor<scalar_t, 2> p_block_o = O[cur_Bi];

	scalar_t block_q[Q_BLOCK_SIZE][C_SIZE];
	scalar_t block_k[KV_BLOCK_SIZE][C_SIZE];
	scalar_t block_v[KV_BLOCK_SIZE][C_SIZE];
	scalar_t block_m[Q_BLOCK_SIZE][KV_BLOCK_SIZE];
	scalar_t block_o[Q_BLOCK_SIZE][C_SIZE];

	copy_arr<true>((scalar_t*)block_q, p_block_q);
	copy_arr<true>((scalar_t*)block_k, p_block_k);
	copy_arr<true>((scalar_t*)block_v, p_block_v);
	copy_arr<true>((scalar_t*)block_m, p_block_m);

	linear_attention_forward(
		block_q, block_k, block_v, block_m, block_o
	);

	copy_arr<false>((scalar_t*)block_o, p_block_o);
}


template <typename scalar_t>
__global__ void test_main_func_bw(
	at::PackedTensorAccessor<scalar_t, 3> Q,
	at::PackedTensorAccessor<scalar_t, 3> K,
	at::PackedTensorAccessor<scalar_t, 3> V,
	at::PackedTensorAccessor<scalar_t, 3> M,
	at::PackedTensorAccessor<scalar_t, 3> dO,
	//
	at::PackedTensorAccessor<scalar_t, 3> dQ,
	at::PackedTensorAccessor<scalar_t, 3> dK,
	at::PackedTensorAccessor<scalar_t, 3> dV,
	at::PackedTensorAccessor<scalar_t, 3> dM
) {
	// 矩阵乘法，用来处理 A @ v
	// pM1 [L1, C] ; pM2 [L2, C] ; pO [L1, L2]
	int cur_Bi = blockIdx.x;
	
	// printf("g%d ", cur_Bi);

	// printf("B%d H%d Q%d K%d\n", cur_Bi, cur_Hi, cur_Qi, cur_Ki);
	at::TensorAccessor<scalar_t, 2> p_block_q = Q[cur_Bi];
	at::TensorAccessor<scalar_t, 2> p_block_k = K[cur_Bi];
	at::TensorAccessor<scalar_t, 2> p_block_v = V[cur_Bi];
	at::TensorAccessor<scalar_t, 2> p_block_m = M[cur_Bi];
	at::TensorAccessor<scalar_t, 2> p_block_do = dO[cur_Bi];
	//
	at::TensorAccessor<scalar_t, 2> p_block_dq = dQ[cur_Bi];
	at::TensorAccessor<scalar_t, 2> p_block_dk = dK[cur_Bi];
	at::TensorAccessor<scalar_t, 2> p_block_dv = dV[cur_Bi];
	at::TensorAccessor<scalar_t, 2> p_block_dm = dM[cur_Bi];

	// fast data
	scalar_t block_q[Q_BLOCK_SIZE][C_SIZE];
	scalar_t block_k[KV_BLOCK_SIZE][C_SIZE];
	scalar_t block_v[KV_BLOCK_SIZE][C_SIZE];
	scalar_t block_m[Q_BLOCK_SIZE][KV_BLOCK_SIZE];
	scalar_t block_do[Q_BLOCK_SIZE][C_SIZE];

	scalar_t block_dq[Q_BLOCK_SIZE][C_SIZE];
	scalar_t block_dk[KV_BLOCK_SIZE][C_SIZE];
	scalar_t block_dv[KV_BLOCK_SIZE][C_SIZE];
	scalar_t block_dm[Q_BLOCK_SIZE][KV_BLOCK_SIZE];
	//
	copy_arr<true>((scalar_t*)block_q, p_block_q);
	copy_arr<true>((scalar_t*)block_k, p_block_k);
	copy_arr<true>((scalar_t*)block_v, p_block_v);
	copy_arr<true>((scalar_t*)block_m, p_block_m);
	copy_arr<true>((scalar_t*)block_do, p_block_do);

	linear_attention_backward(
		block_q, block_k, block_v, block_m, block_do,
		block_dq, block_dk, block_dv, block_dm
	);

	copy_arr<false>((scalar_t*)block_dm, p_block_dm);
	copy_arr<false>((scalar_t*)block_dq, p_block_dq);
	copy_arr<false>((scalar_t*)block_dk, p_block_dk);
	copy_arr<false>((scalar_t*)block_dv, p_block_dv);
}


// 快速测试
void cuda_fast_block_fw(
	at::Tensor& Q,
	at::Tensor& K,
	at::Tensor& V,
	at::Tensor& M,
	at::Tensor& O
){
	O.zero_();
	// B L C
	dim3 grid(Q.size(0));
	dim3 block(1);

	using scalar_t = float;
	test_main_func<scalar_t>
	<<<grid, block>>>
	(
		Q.packed_accessor<scalar_t, 3>(),
		K.packed_accessor<scalar_t, 3>(),
		V.packed_accessor<scalar_t, 3>(),
		M.packed_accessor<scalar_t, 3>(),
		O.packed_accessor<scalar_t, 3>()
	);
}


// 快速测试2
void cuda_fast_block_bw(
	//
	at::Tensor& Q,
	at::Tensor& K,
	at::Tensor& V,
	at::Tensor& M,
	at::Tensor& dO,
	//
	at::Tensor& dQ,
	at::Tensor& dK,
	at::Tensor& dV,
	at::Tensor& dM
){
	dQ.zero_();
	dK.zero_();
	dV.zero_();
	dM.zero_();
	// B L C
	dim3 grid(Q.size(0));
	dim3 block(1);

	using scalar_t = float;
	test_main_func_bw<scalar_t>
	<<<grid, block>>>
	(
		Q.packed_accessor<scalar_t, 3>(),
		K.packed_accessor<scalar_t, 3>(),
		V.packed_accessor<scalar_t, 3>(),
		M.packed_accessor<scalar_t, 3>(),
		dO.packed_accessor<scalar_t, 3>(),
		//
		dQ.packed_accessor<scalar_t, 3>(),
		dK.packed_accessor<scalar_t, 3>(),
		dV.packed_accessor<scalar_t, 3>(),
		dM.packed_accessor<scalar_t, 3>()
	);
}

// 主前向函数
void forward(
	// in
	at::Tensor& Q,
	at::Tensor& K,
	at::Tensor& V,
	at::Tensor& M,
	// out
	at::Tensor& O
){
	/*
		Q [B, H, qL, C]
		K [B, H, kL, C]
		V [B, H, kL, C]
		M [B, H, qL, kL]
		O [B, H, qL, C]
	*/

	// // test
	// cudaDeviceSynchronize();
	// auto tt1 = at::rand({200, 100}, at::device(at::kCUDA));
	// auto tto1 = at::zeros({200, 100}, at::device(at::kCUDA));
	// auto tto2 = at::zeros({200, 100}, at::device(at::kCUDA));

	// for (int i=0; i<200; ++i)
	// 	tto1 += tt1;

	// omp_set_num_threads(20);
	// #pragma omp parallel for schedule(static)
	// for (int i=0; i<200; ++i)
	// 	tto2 += tt1;
	
	// cudaDeviceSynchronize();
	// printf("diff %f \n", (tto1-tto2).abs().max());
	// //

	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	using scalar_t = float;

	// 在外面使用 expand 支持 mask 的 B == 1 和 H == 1 ，里面不写支持
	// 在外面使用 expand 支持 k_head == 1 ，里面不写支持

	int B=Q.size(0), H=Q.size(1), qL=Q.size(2), C=Q.size(3);
	int kL=K.size(2);

	TORCH_CHECK(qL % Q_BLOCK_SIZE == 0, "Bad");
	TORCH_CHECK(kL % KV_BLOCK_SIZE == 0, "Bad");

	int qN = qL / Q_BLOCK_SIZE;
	int kN = kL / KV_BLOCK_SIZE;

	auto Q_ = Q.view({B, H, qN, Q_BLOCK_SIZE, C});
	auto K_ = K.view({B, H, kN, KV_BLOCK_SIZE, C});
	auto V_ = V.view({B, H, kN, KV_BLOCK_SIZE, C});
	auto M_ = M.view({B, H, qN, Q_BLOCK_SIZE, kN, KV_BLOCK_SIZE}).swapdims(3, 4);
	auto O_ = O.view({B, H, qN, Q_BLOCK_SIZE, C});

	// printf("Q shape %lld %lld %lld %lld %lld\n", Q_.size(0), Q_.size(1), Q_.size(2), Q_.size(3), Q_.size(4));
	// printf("K shape %lld %lld %lld %lld %lld\n", K_.size(0), K_.size(1), K_.size(2), K_.size(3), K_.size(4));
	// printf("O shape %lld %lld %lld %lld %lld\n", O_.size(0), O_.size(1), O_.size(2), O_.size(3), O_.size(4));

	dim3 grid(B*H, qN);
	dim3 block(kN);

	// printf("B*H=%d qN=%d kN=%d", B*H, qN, kN);

	linear_attention_forward_kernel<scalar_t>
	<<<grid, block, 0, stream>>>
	(
		Q_.packed_accessor<scalar_t, 5>(),
		K_.packed_accessor<scalar_t, 5>(),
		V_.packed_accessor<scalar_t, 5>(),
		M_.packed_accessor<scalar_t, 6>(),
		O_.packed_accessor<scalar_t, 5>(),
		B,
		H,
		qL,
		kL,
		qN,
		kN,
		C
	);
}

// 主反向函数
void backward(
	// in
	at::Tensor& Q,
	at::Tensor& K,
	at::Tensor& V,
	at::Tensor& M,
	at::Tensor& dO,
	// out
	at::Tensor& dQ,
	at::Tensor& dK,
	at::Tensor& dV,
	at::Tensor& dM
){
	/*
		Q [B, H, qL, C]
		K [B, H, kL, C]
		V [B, H, kL, C]
		M [B, H, qL, kL]

		dO [B, H, qL, C]
		dQ [B, H, qL, C]
		dK [B, H, kL, C]
		dV [B, H, kL, C]
		dM [B, H, qL, kL]
	*/
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	using scalar_t = float;

	// 在外面使用 expand 支持 mask 的 B == 1 和 H == 1 ，里面不写支持
	// 在外面使用 expand 支持 k_head == 1 ，里面不写支持

	int B=Q.size(0), H=Q.size(1), qL=Q.size(2), C=Q.size(3);
	int kL=K.size(2);

	dQ.zero_();
	dK.zero_();
	dV.zero_();
	dM.zero_();

	TORCH_CHECK(qL % Q_BLOCK_SIZE == 0, "Bad qL");
	TORCH_CHECK(kL % KV_BLOCK_SIZE == 0, "Bad kL");
	TORCH_CHECK(kL % KV_BLOCK_SIZE == 0, "Bad");

	int qN = qL / Q_BLOCK_SIZE;
	int kN = kL / KV_BLOCK_SIZE;

	auto Q_ = Q.view({B, H, qN, Q_BLOCK_SIZE, C});
	auto K_ = K.view({B, H, kN, KV_BLOCK_SIZE, C});
	auto V_ = V.view({B, H, kN, KV_BLOCK_SIZE, C});
	auto M_ = M.view({B, H, qN, Q_BLOCK_SIZE, kN, KV_BLOCK_SIZE}).swapdims(3, 4);
	auto dO_ = dO.view({B, H, qN, Q_BLOCK_SIZE, C});

	auto dQ_ = dQ.view({B, H, qN, Q_BLOCK_SIZE, C});
	auto dK_ = dK.view({B, H, kN, KV_BLOCK_SIZE, C});
	auto dV_ = dV.view({B, H, kN, KV_BLOCK_SIZE, C});
	auto dM_ = dM.view({B, H, qN, Q_BLOCK_SIZE, kN, KV_BLOCK_SIZE}).swapdims(3, 4);

	// printf("Q shape %lld %lld %lld %lld %lld\n", Q_.size(0), Q_.size(1), Q_.size(2), Q_.size(3), Q_.size(4));
	// printf("K shape %lld %lld %lld %lld %lld\n", K_.size(0), K_.size(1), K_.size(2), K_.size(3), K_.size(4));
	// printf("O shape %lld %lld %lld %lld %lld\n", O_.size(0), O_.size(1), O_.size(2), O_.size(3), O_.size(4));

	dim3 grid(B*H, kN);
	dim3 block(qN);

	linear_attention_backward_kernel<scalar_t>
	<<<grid, block, 0, stream>>>
	(
		Q_.packed_accessor<scalar_t, 5>(),
		K_.packed_accessor<scalar_t, 5>(),
		V_.packed_accessor<scalar_t, 5>(),
		M_.packed_accessor<scalar_t, 6>(),
		dO_.packed_accessor<scalar_t, 5>(),
		//
		dQ_.packed_accessor<scalar_t, 5>(),
		dK_.packed_accessor<scalar_t, 5>(),
		dV_.packed_accessor<scalar_t, 5>(),
		dM_.packed_accessor<scalar_t, 6>(),
		//
		B,
		H,
		qL,
		kL,
		qN,
		kN,
		C
	);
}
