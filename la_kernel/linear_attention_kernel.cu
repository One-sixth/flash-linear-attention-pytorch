/*
带_par的函数是线程块版本，不带的是单线程版本。单线程版本用来验证线程块版本是否正确运行
*/


#include <iostream>
#include <cmath>
#include <vector>
#include <torch/all.h>
#include <ATen/ATen.h>
#include <torch/library.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAGraphsUtils.cuh>

/**********************************************************************/

#ifndef _C_SIZE
#define _C_SIZE 32
#endif

#ifndef _Q_BLOCK_SIZE
#define _Q_BLOCK_SIZE 32
#endif

#ifndef _KV_BLOCK_SIZE
#define _KV_BLOCK_SIZE 32
#endif

#ifndef _DTYPE
#define _DTYPE float
#endif

constexpr int Q_BLOCK_SIZE = _Q_BLOCK_SIZE;
constexpr int KV_BLOCK_SIZE = _KV_BLOCK_SIZE;
constexpr int C_SIZE = _C_SIZE;

// 在Gtx1070上，该线程块设置似乎最优
constexpr int FORWARD_THREAD_H = 4;
constexpr int FORWARD_THREAD_W = 32;
constexpr int BACKWARD_THREAD_H = 4;
constexpr int BACKWARD_THREAD_W = 32;

/**********************************************************************/
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor");


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
	// 矩阵乘法
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


template <
	int m1_row, int m1_col, bool m1_transpose,
	int m2_row, int m2_col, bool m2_transpose,
	int o_row, int o_col,
	typename scalar_t
>
__device__ inline void my_matmul_par(
	scalar_t pM1[m1_row][m1_col],
	scalar_t pM2[m2_row][m2_col],
	scalar_t pMO[o_row][o_col],
	int cur_thread_y,
	int cur_thread_x,
	int thread_block_h,
	int thread_block_w
){
	// 矩阵乘法
	constexpr int L1 = m1_transpose ? m1_col : m1_row;
	constexpr int C = m1_transpose ? m1_row : m1_col;
	constexpr int L2 = m2_transpose ? m2_row : m2_col;

	static_assert(L1 == o_row, "L1 must be == o_row");
	static_assert(L2 == o_col, "L2 must be == o_col");

	for (int i=cur_thread_y; i<L1; i+=thread_block_h)
		for (int j=cur_thread_x; j<L2; j+=thread_block_w)
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
__device__ inline void mul_arr(
	scalar_t src1[],
	scalar_t src2[],
	scalar_t dst[]
){
	// dst = src1 * src2
	for (int i=0; i<n; i++)
		dst[i] = src1[i] * src2[i];
}


template <int row, int col, typename scalar_t>
__device__ inline void mul_arr_par(
	scalar_t src1[row][col],
	scalar_t src2[row][col],
	scalar_t dst[row][col],
	int cur_thread_y,
	int cur_thread_x,
	int thread_block_h,
	int thread_block_w
){
	// dst = src1 * src2
	for (int i=cur_thread_y; i<row; i+=thread_block_h)
		for (int j=cur_thread_x; j<col; j+=thread_block_w)
			dst[i][j] = src1[i][j] * src2[i][j];
}


template <int n, typename scalar_t>
__device__ inline void add_arr(
	scalar_t src1[],
	scalar_t src2[],
	scalar_t dst[]
){
	// dst = src1 + src2
	for (int i=0; i<n; i++)
		dst[i] = src1[i] + src2[i];
}


template <int row, int col, typename scalar_t>
__device__ inline void add_arr_par(
	scalar_t src1[row][col],
	scalar_t src2[row][col],
	scalar_t dst[row][col],
	int cur_thread_y,
	int cur_thread_x,
	int thread_block_h,
	int thread_block_w
){
	// dst = src1 + src2
	for (int i=cur_thread_y; i<row; i+=thread_block_h)
		for (int j=cur_thread_x; j<col; j+=thread_block_w)
			dst[i][j] = src1[i][j] + src2[i][j];
}


template <bool dst_to_src, typename scalar_t>
__device__ inline void add_arr(
	scalar_t src[],
	torch::TensorAccessor<scalar_t, 2> dst
){
	// dst += src
	const int row = dst.size(0);
	const int col = dst.size(1);

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
__device__ inline void atomic_add_arr(
	scalar_t src[],
	scalar_t dst[]
){
	// dst += src
	for (int i=0; i<n; i++)
		gpuAtomicAdd(&dst[i], src[i]);
}


template <bool dst_to_src, typename scalar_t>
__device__ inline void atomic_add_arr(
	scalar_t src[],
	torch::TensorAccessor<scalar_t, 2> dst
){
	// dst += src
	const int row = dst.size(0);
	const int col = dst.size(1);

	for (int cur_row=0; cur_row<row; cur_row++)
		for (int cur_col=0; cur_col<col; cur_col++)
		{
			if (dst_to_src)
				gpuAtomicAdd(&src[cur_row*col+cur_col], dst[cur_row][cur_col]);
			else
				gpuAtomicAdd(&dst[cur_row][cur_col], src[cur_row*col+cur_col]);
		}
}


template <int row, int col, bool dst_to_src, typename scalar_t>
__device__ inline void atomic_add_arr_par(
	scalar_t src[row][col],
	scalar_t dst[row][col],
	int cur_thread_y,
	int cur_thread_x,
	int thread_block_h,
	int thread_block_w
){
	// dst += src
	for (int i=cur_thread_y; i<row; i+=thread_block_h)
		for (int j=cur_thread_x; j<col; j+=thread_block_w)
		{
			if (dst_to_src)
				gpuAtomicAdd(&src[i][j], dst[i][j]);
			else
				gpuAtomicAdd(&dst[i][j], src[i][j]);
		}
}


template <int row, int col, bool dst_to_src, typename scalar_t>
__device__ inline void atomic_add_arr_par(
	scalar_t src[row][col],
	torch::TensorAccessor<scalar_t, 2> dst,
	int cur_thread_y,
	int cur_thread_x,
	int thread_block_h,
	int thread_block_w
){
	// dst += src
	for (int i=cur_thread_y; i<row; i+=thread_block_h)
		for (int j=cur_thread_x; j<col; j+=thread_block_w)
		{
			if (dst_to_src)
				gpuAtomicAdd(&src[i][j], dst[i][j]);
			else
				gpuAtomicAdd(&dst[i][j], src[i][j]);
		}
}


template <int n, typename scalar_t>
__device__ inline void copy_arr(
	scalar_t src[],
	scalar_t dst[]
){
	// dst = src
	for (int i=0; i<n; i++)
		dst[i] = src[i];
}


template <int row, int col, typename scalar_t>
__device__ inline void copy_arr_par(
	scalar_t src[row][col],
	scalar_t dst[row][col],
	int cur_thread_y,
	int cur_thread_x,
	int thread_block_h,
	int thread_block_w
){
	// dst = src
	for (int i=cur_thread_y; i<row; i+=thread_block_h)
		for (int j=cur_thread_x; j<col; j+=thread_block_w)
			dst[i][j] = src[i][j];
}


template <bool dst_to_src, typename scalar_t>
__device__ inline void copy_arr(
	scalar_t src[],
	torch::TensorAccessor<scalar_t, 2> dst
){
	// dst = src
	const int row = dst.size(0);
	const int col = dst.size(1);

	for (int cur_row=0; cur_row<row; cur_row++)
		for (int cur_col=0; cur_col<col; cur_col++)
		{
			if (dst_to_src)
				src[cur_row*col+cur_col] = dst[cur_row][cur_col];
			else
				dst[cur_row][cur_col] = src[cur_row*col+cur_col];
		}
}


template <int row, int col, bool dst_to_src, typename scalar_t>
__device__ inline void copy_arr_par(
	scalar_t src[row][col],
	torch::TensorAccessor<scalar_t, 2> dst,
	int cur_thread_y,
	int cur_thread_x,
	int thread_block_h,
	int thread_block_w
){
	// dst = src
	for (int i=cur_thread_y; i<row; i+=thread_block_h)
		for (int j=cur_thread_x; j<col; j+=thread_block_w)
		{
			if (dst_to_src)
				src[i][j] = dst[i][j];
			else
				dst[i][j] = src[i][j];
		}
}


template <int row, int col, typename scalar_t>
__device__ inline void zero_arr_par(
	scalar_t dst[row][col],
	int cur_thread_y,
	int cur_thread_x,
	int thread_block_h,
	int thread_block_w
){
	// dst = src
	for (int i=cur_thread_y; i<row; i+=thread_block_h)
		for (int j=cur_thread_x; j<col; j+=thread_block_w)
			dst[i][j] = 0;
}


template<typename scalar_t>
__global__ void linear_attention_forward_kernel(
	torch::PackedTensorAccessor<scalar_t, 5> Q,
	torch::PackedTensorAccessor<scalar_t, 5> K,
	torch::PackedTensorAccessor<scalar_t, 5> V,
	torch::PackedTensorAccessor<scalar_t, 6> M,
	torch::PackedTensorAccessor<scalar_t, 5> O,
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
	int cur_Ki = blockIdx.z;
	int cur_thread_y = threadIdx.y;
	int cur_thread_x = threadIdx.x;
	int thread_block_h = blockDim.y;
	int thread_block_w = blockDim.x;

	// printf("B%d H%d Q%d K%d Y%d X%d H%d W%d\n", cur_Bi, cur_Hi, cur_Qi, cur_Ki, cur_thread_y, cur_thread_x, thread_block_h, thread_block_w);

	// 获得当前线程的分块数据
	torch::TensorAccessor<scalar_t, 2> p_block_q = Q[cur_Bi][cur_Hi][cur_Qi];
	torch::TensorAccessor<scalar_t, 2> p_block_k = K[cur_Bi][cur_Hi][cur_Ki];
	torch::TensorAccessor<scalar_t, 2> p_block_v = V[cur_Bi][cur_Hi][cur_Ki];
	torch::TensorAccessor<scalar_t, 2> p_block_m = M[cur_Bi][cur_Hi][cur_Qi][cur_Ki];
	torch::TensorAccessor<scalar_t, 2> p_block_o = O[cur_Bi][cur_Hi][cur_Qi];

	// in data
	__shared__ scalar_t block_q[Q_BLOCK_SIZE][C_SIZE];
	__shared__ scalar_t block_k[KV_BLOCK_SIZE][C_SIZE];
	__shared__ scalar_t block_v[KV_BLOCK_SIZE][C_SIZE];
	__shared__ scalar_t block_m[Q_BLOCK_SIZE][KV_BLOCK_SIZE];

	// temp data
	__shared__ scalar_t A[Q_BLOCK_SIZE][KV_BLOCK_SIZE];

	// out data
	// 复用 block_q 的空间给 block_o
	auto& block_o = block_q;

	// load data
	copy_arr_par<Q_BLOCK_SIZE, C_SIZE, true>(block_q, p_block_q, cur_thread_y, cur_thread_x, thread_block_h, thread_block_w);
	copy_arr_par<KV_BLOCK_SIZE, C_SIZE, true>(block_k, p_block_k, cur_thread_y, cur_thread_x, thread_block_h, thread_block_w);
	copy_arr_par<KV_BLOCK_SIZE, C_SIZE, true>(block_v, p_block_v, cur_thread_y, cur_thread_x, thread_block_h, thread_block_w);
	copy_arr_par<Q_BLOCK_SIZE, KV_BLOCK_SIZE, true>(block_m, p_block_m, cur_thread_y, cur_thread_x, thread_block_h, thread_block_w);
	__syncthreads();

	/***********/
	// 线性注意力-前向传播函数
	// A = Q @ K.transpose(-1, -2)
	my_matmul_par<
		Q_BLOCK_SIZE, C_SIZE, false,
		KV_BLOCK_SIZE, C_SIZE, true,
		Q_BLOCK_SIZE, KV_BLOCK_SIZE
	>(block_q, block_k, A, cur_thread_y, cur_thread_x, thread_block_h, thread_block_w);
	__syncthreads();

	// A 在后面没用了，这里让Am复用A的空间
	auto &Am = A;
	// Am = A * M
	mul_arr_par<Q_BLOCK_SIZE, KV_BLOCK_SIZE>(A, block_m, Am, cur_thread_y, cur_thread_x, thread_block_h, thread_block_w);
	__syncthreads();

	// O = Am @ V
	my_matmul_par<
		Q_BLOCK_SIZE, KV_BLOCK_SIZE, false,
		KV_BLOCK_SIZE, C_SIZE, false,
		Q_BLOCK_SIZE, C_SIZE
	>(A, block_v, block_o, cur_thread_y, cur_thread_x, thread_block_h, thread_block_w);
	__syncthreads();
	/***********/

	atomic_add_arr_par<Q_BLOCK_SIZE, C_SIZE, false>(block_o, p_block_o, cur_thread_y, cur_thread_x, thread_block_h, thread_block_w);
}


template<typename scalar_t>
__global__ void linear_attention_backward_kernel(
	// in
	torch::PackedTensorAccessor<scalar_t, 5> Q,
	torch::PackedTensorAccessor<scalar_t, 5> K,
	torch::PackedTensorAccessor<scalar_t, 5> V,
	torch::PackedTensorAccessor<scalar_t, 6> M,
	torch::PackedTensorAccessor<scalar_t, 5> dO,
	// out
	torch::PackedTensorAccessor<scalar_t, 5> dQ,
	torch::PackedTensorAccessor<scalar_t, 5> dK,
	torch::PackedTensorAccessor<scalar_t, 5> dV,
	torch::PackedTensorAccessor<scalar_t, 6> dM,
	// other
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
	int cur_Ki = blockIdx.y;
	int cur_Qi = blockIdx.z;
	int cur_thread_y = threadIdx.y;
	int cur_thread_x = threadIdx.x;
	int thread_block_h = blockDim.y;
	int thread_block_w = blockDim.x;

	// printf("B%d H%d Q%d K%d Y%d X%d H%d W%d\n", cur_Bi, cur_Hi, cur_Qi, cur_Ki, cur_thread_y, cur_thread_x, thread_block_h, thread_block_w);

	// 获得当前线程的分块数据
	torch::TensorAccessor<scalar_t, 2> p_block_q = Q[cur_Bi][cur_Hi][cur_Qi];
	torch::TensorAccessor<scalar_t, 2> p_block_k = K[cur_Bi][cur_Hi][cur_Ki];
	torch::TensorAccessor<scalar_t, 2> p_block_v = V[cur_Bi][cur_Hi][cur_Ki];
	torch::TensorAccessor<scalar_t, 2> p_block_m = M[cur_Bi][cur_Hi][cur_Qi][cur_Ki];
	torch::TensorAccessor<scalar_t, 2> p_block_do = dO[cur_Bi][cur_Hi][cur_Qi];
	//
	torch::TensorAccessor<scalar_t, 2> p_block_dq = dQ[cur_Bi][cur_Hi][cur_Qi];
	torch::TensorAccessor<scalar_t, 2> p_block_dk = dK[cur_Bi][cur_Hi][cur_Ki];
	torch::TensorAccessor<scalar_t, 2> p_block_dv = dV[cur_Bi][cur_Hi][cur_Ki];
	torch::TensorAccessor<scalar_t, 2> p_block_dm = dM[cur_Bi][cur_Hi][cur_Qi][cur_Ki];

	// in data
	__shared__ scalar_t block_q[Q_BLOCK_SIZE][C_SIZE];
	__shared__ scalar_t block_k[KV_BLOCK_SIZE][C_SIZE];
	__shared__ scalar_t block_v[KV_BLOCK_SIZE][C_SIZE];
	__shared__ scalar_t block_m[Q_BLOCK_SIZE][KV_BLOCK_SIZE];
	__shared__ scalar_t block_do[Q_BLOCK_SIZE][C_SIZE];

	// temp data
	__shared__ scalar_t A[Q_BLOCK_SIZE][KV_BLOCK_SIZE];
	__shared__ scalar_t dAm[Q_BLOCK_SIZE][KV_BLOCK_SIZE];
	__shared__ scalar_t dA[Q_BLOCK_SIZE][KV_BLOCK_SIZE];
	// A后面没用了，重用A的空间给Am
	auto& Am = A;

	// out data
	// 这里也是复用 block_q 作为 block_dq，节省空间
	auto& block_dq = block_q;
	// // 复用 block_v
	auto& block_dk = block_v;
	// // 复用 block_v
	auto& block_dv = block_v;
	// __shared__ scalar_t block_dk[KV_BLOCK_SIZE][C_SIZE];
	// __shared__ scalar_t block_dv[KV_BLOCK_SIZE][C_SIZE];
	// 复用 block_m
	auto& block_dm = dA;
	// __shared__ scalar_t block_dm[Q_BLOCK_SIZE][KV_BLOCK_SIZE];

	// load data
	copy_arr_par<Q_BLOCK_SIZE, C_SIZE, true>(block_q, p_block_q, cur_thread_y, cur_thread_x, thread_block_h, thread_block_w);
	copy_arr_par<KV_BLOCK_SIZE, C_SIZE, true>(block_k, p_block_k, cur_thread_y, cur_thread_x, thread_block_h, thread_block_w);
	copy_arr_par<KV_BLOCK_SIZE, C_SIZE, true>(block_v, p_block_v, cur_thread_y, cur_thread_x, thread_block_h, thread_block_w);
	copy_arr_par<Q_BLOCK_SIZE, KV_BLOCK_SIZE, true>(block_m, p_block_m, cur_thread_y, cur_thread_x, thread_block_h, thread_block_w);
	copy_arr_par<Q_BLOCK_SIZE, C_SIZE, true>(block_do, p_block_do, cur_thread_y, cur_thread_x, thread_block_h, thread_block_w);
	__syncthreads();

	/********************/
	// 线性注意力-反向传播函数
	// 提前写入全局内存部分语句，进一步减少共享内存消耗
	// A = Q @ K.transpose(-1, -2)
	my_matmul_par<
		Q_BLOCK_SIZE, C_SIZE, false,
		KV_BLOCK_SIZE, C_SIZE, true,
		Q_BLOCK_SIZE, KV_BLOCK_SIZE
	>(block_q, block_k, A, cur_thread_y, cur_thread_x, thread_block_h, thread_block_w);
	// dAm = dO @ V.transpose(-1, -2)
	my_matmul_par<
		Q_BLOCK_SIZE, C_SIZE, false,
		KV_BLOCK_SIZE, C_SIZE, true,
		Q_BLOCK_SIZE, KV_BLOCK_SIZE
	>(block_do, block_v, dAm, cur_thread_y, cur_thread_x, thread_block_h, thread_block_w);
	__syncthreads();

	// dM = dAm * A
	mul_arr_par<Q_BLOCK_SIZE, KV_BLOCK_SIZE>(dAm, A, block_dm, cur_thread_y, cur_thread_x, thread_block_h, thread_block_w);
	__syncthreads();
	// 提前写入 block_dm ，然后省出空间给 dA 使用
	atomic_add_arr_par<Q_BLOCK_SIZE, KV_BLOCK_SIZE, false>(block_dm, p_block_dm, cur_thread_y, cur_thread_x, thread_block_h, thread_block_w);
	__syncthreads();

	// dA = dAm * M
	mul_arr_par<Q_BLOCK_SIZE, KV_BLOCK_SIZE>(dAm, block_m, dA, cur_thread_y, cur_thread_x, thread_block_h, thread_block_w);
	// A后面没用了，重用A ，现在 A的空间由Am使用
	// Am = A * M
	mul_arr_par<Q_BLOCK_SIZE, KV_BLOCK_SIZE>(A, block_m, Am, cur_thread_y, cur_thread_x, thread_block_h, thread_block_w);
	__syncthreads();

	// dV = Am.transpose(-1, -2) @ dO
	my_matmul_par<
		Q_BLOCK_SIZE, KV_BLOCK_SIZE, true,
		Q_BLOCK_SIZE, C_SIZE, false,
		KV_BLOCK_SIZE, C_SIZE
	>(Am, block_do, block_dv, cur_thread_y, cur_thread_x, thread_block_h, thread_block_w);
	__syncthreads();
	// 提前写入 block_dv，然后省出来给 block_dk 用
	atomic_add_arr_par<KV_BLOCK_SIZE, C_SIZE, false>(block_dv, p_block_dv, cur_thread_y, cur_thread_x, thread_block_h, thread_block_w);
	__syncthreads();

	// dK = dA.transpose(-1, -2) @ Q
	my_matmul_par<
		Q_BLOCK_SIZE, KV_BLOCK_SIZE, true,
		Q_BLOCK_SIZE, C_SIZE, false,
		KV_BLOCK_SIZE, C_SIZE
	>(dA, block_q, block_dk, cur_thread_y, cur_thread_x, thread_block_h, thread_block_w);
	__syncthreads();
	// dQ = dA @ K
	my_matmul_par<
		Q_BLOCK_SIZE, KV_BLOCK_SIZE, false,
		KV_BLOCK_SIZE, C_SIZE, false,
		Q_BLOCK_SIZE, C_SIZE
	>(dA, block_k, block_dq, cur_thread_y, cur_thread_x, thread_block_h, thread_block_w);
	__syncthreads();

	/********************/

	atomic_add_arr_par<Q_BLOCK_SIZE, C_SIZE, false>(block_dq, p_block_dq, cur_thread_y, cur_thread_x, thread_block_h, thread_block_w);
	atomic_add_arr_par<KV_BLOCK_SIZE, C_SIZE, false>(block_dk, p_block_dk, cur_thread_y, cur_thread_x, thread_block_h, thread_block_w);
	// 已被提前
	// atomic_add_arr_par<KV_BLOCK_SIZE, C_SIZE, false>(block_dv, p_block_dv, cur_thread_y, cur_thread_x, thread_block_h, thread_block_w);
	// 已被提前
	// atomic_add_arr_par<Q_BLOCK_SIZE, KV_BLOCK_SIZE, false>(block_dm, p_block_dm, cur_thread_y, cur_thread_x, thread_block_h, thread_block_w);
}


// 主前向函数
void forward(
	// in
	torch::Tensor& Q,
	torch::Tensor& K,
	torch::Tensor& V,
	torch::Tensor& M,
	// out
	torch::Tensor& O
){
	/*
		Q [B, H, qL, C]
		K [B, H, kL, C]
		V [B, H, kL, C]
		M [B, H, qL, kL]
		O [B, H, qL, C]
	*/

	cudaStream_t stream = at::cuda::getCurrentCUDAStream();

	// 在外面使用 expand 支持 mask 的 B == 1 和 H == 1 ，里面不写支持
	// 在外面使用 expand 支持 k_head == 1 ，里面不写支持

	int B=Q.size(0), H=Q.size(1), qL=Q.size(2), C=Q.size(3);
	int kL=K.size(2);

	TORCH_CHECK(qL % Q_BLOCK_SIZE == 0, "qL must be a multiple of Q_BLOCK_SIZE.");
	TORCH_CHECK(kL % KV_BLOCK_SIZE == 0, "kL must be a multiple of KV_BLOCK_SIZE.");
	CHECK_CUDA(Q);
	CHECK_CUDA(K);
	CHECK_CUDA(V);
	CHECK_CUDA(M);
	CHECK_CUDA(O);

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

	dim3 grid(B*H, qN, kN);
	dim3 block(FORWARD_THREAD_H, FORWARD_THREAD_W);

	// printf("B*H=%d qN=%d kN=%d", B*H, qN, kN);

	using scalar_t = _DTYPE;
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

	// 不使用自动生成内核，因为 double 会占用更多的 共享内存，然后编译不通过
	// 使用 AT_DISPATCH_FLOATING_TYPES 宏自动生成支持不同类型的内核
	// AT_DISPATCH_FLOATING_TYPES(Q.type(), "linear attention forward", ([&] {
	// 	linear_attention_forward_kernel<scalar_t><<<grid, block, 0, stream>>>
	// 	(
	// 		Q_.packed_accessor<scalar_t, 5>(),
	// 		K_.packed_accessor<scalar_t, 5>(),
	// 		V_.packed_accessor<scalar_t, 5>(),
	// 		M_.packed_accessor<scalar_t, 6>(),
	// 		O_.packed_accessor<scalar_t, 5>(),
	// 		B,
	// 		H,
	// 		qL,
	// 		kL,
	// 		qN,
	// 		kN,
	// 		C
	// 	);
	// }));
	
	// THCudaCheck(cudaGetLastError());
}

// 主反向函数
void backward(
	// in
	torch::Tensor& Q,
	torch::Tensor& K,
	torch::Tensor& V,
	torch::Tensor& M,
	torch::Tensor& dO,
	// out
	torch::Tensor& dQ,
	torch::Tensor& dK,
	torch::Tensor& dV,
	torch::Tensor& dM
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

	// 在外面使用 expand 支持 mask 的 B == 1 和 H == 1 ，里面不写支持
	// 在外面使用 expand 支持 k_head == 1 ，里面不写支持

	int B=Q.size(0), H=Q.size(1), qL=Q.size(2), C=Q.size(3);
	int kL=K.size(2);

	dQ.zero_();
	dK.zero_();
	dV.zero_();
	dM.zero_();

	// 用C++写输入检查太麻烦了，放在外面进行，这里就写简单的
	TORCH_CHECK(qL % Q_BLOCK_SIZE == 0, "qL must be a multiple of Q_BLOCK_SIZE.");
	TORCH_CHECK(kL % KV_BLOCK_SIZE == 0, "kL must be a multiple of KV_BLOCK_SIZE.");

	CHECK_CUDA(Q);
	CHECK_CUDA(K);
	CHECK_CUDA(V);
	CHECK_CUDA(M);
	CHECK_CUDA(dO);
	CHECK_CUDA(dQ);
	CHECK_CUDA(dK);
	CHECK_CUDA(dV);
	CHECK_CUDA(dM);

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

	// 让外循环是K，内循环是Q，可以减少一些原子加法写入冲突，加速
	dim3 grid(B*H, kN, qN);
	dim3 block(BACKWARD_THREAD_H, BACKWARD_THREAD_W);

	using scalar_t = _DTYPE;
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

	// 不使用自动生成内核，因为 double 会占用更多的 共享内存，然后编译不通过
	// 使用 AT_DISPATCH_FLOATING_TYPES 宏自动生成支持不同类型的内核
	// AT_DISPATCH_FLOATING_TYPES(Q.type(), "linear attention backward", ([&] {
	// 	linear_attention_backward_kernel<scalar_t><<<grid, block, 0, stream>>>
	// 	(
	// 		Q_.packed_accessor<scalar_t, 5>(),
	// 		K_.packed_accessor<scalar_t, 5>(),
	// 		V_.packed_accessor<scalar_t, 5>(),
	// 		M_.packed_accessor<scalar_t, 6>(),
	// 		dO_.packed_accessor<scalar_t, 5>(),
	// 		//
	// 		dQ_.packed_accessor<scalar_t, 5>(),
	// 		dK_.packed_accessor<scalar_t, 5>(),
	// 		dV_.packed_accessor<scalar_t, 5>(),
	// 		dM_.packed_accessor<scalar_t, 6>(),
	// 		//
	// 		B,
	// 		H,
	// 		qL,
	// 		kL,
	// 		qN,
	// 		kN,
	// 		C
	// 	);
	// }));

	// THCudaCheck(cudaGetLastError());
}
