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


constexpr int Q_BLOCK_SIZE = 32;
constexpr int KV_BLOCK_SIZE = 32;


template <typename scalar_t, bool m1_transpose, bool m2_transpose, bool out_atomic_add>
__host__ __device__ inline void my_matmul(
	at::TensorAccessor<scalar_t, 2> pM1,
	at::TensorAccessor<scalar_t, 2> pM2,
	at::TensorAccessor<scalar_t, 2> pMO
){
	// 矩阵乘法，m2_transpose 用来处理 Q @ K.T
	// pM1 [L1, C] ; pM2 [L2, C] ; pO [L1, L2]
	int L1 = m1_transpose ? pM1.size(1) : pM1.size(0);
	int C = m1_transpose ? pM1.size(0) : pM1.size(1);

	int L2 = m2_transpose ? pM2.size(0) : pM2.size(1);

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
			if (out_atomic_add)
			{
				// auto ddd = pMO[i][j];
				atomicAdd(&pMO[i][j], s);
				// if (i <= 5 && j <= 5)
				// 	printf("a %f %f %f b\n", ddd, s, pMO[i][j]);
			}
			else
				pMO[i][j] = s;
		}
}


template <typename scalar_t>
__host__ __device__ inline void apply_mul_mask(
	at::TensorAccessor<scalar_t, 2> M,
	at::TensorAccessor<scalar_t, 2> mask
){
	// 对矩阵应用乘法掩码
	// M *= mask
	// M [L, C] ; mask [L, C]
	int L=M.size(0), C=M.size(1);

	for (int i=0; i<L; i++)
		for (int j=0; j<C; j++)
			M[i][j] *= mask[i][j];
}


template <typename scalar_t>
__host__ __device__ inline void apply_add_mask(
	at::TensorAccessor<scalar_t, 2> M,
	at::TensorAccessor<scalar_t, 2> mask
){
	// 对矩阵应用加法掩码
	// M += mask
	// M [L, C] ; mask [L, C]
	int L=M.size(0), C=M.size(1);

	for (int i=0; i<L; i++)
		for (int j=0; j<C; j++)
			M[i][j] += mask[i][j];
}


template <typename scalar_t, bool out_atomic_add>
__host__ __device__ inline void mul_arr(
	at::TensorAccessor<scalar_t, 2> A1,
	at::TensorAccessor<scalar_t, 2> A2,
	at::TensorAccessor<scalar_t, 2> A3
){
	// A3 = A1 * A2
	// M [L, C] ; mask [L, C]
	int L=A1.size(0), C=A1.size(1);

	for (int i=0; i<L; i++)
		for (int j=0; j<C; j++)
		{
			scalar_t s = A1[i][j] * A2[i][j];
			if (out_atomic_add)
				atomicAdd(&A3[i][j], s);
			else
				A3[i][j] = s;
		}
}


template <typename scalar_t, bool out_atomic_add>
__host__ __device__ inline void add_arr(
	at::TensorAccessor<scalar_t, 2> A1,
	at::TensorAccessor<scalar_t, 2> A2,
	at::TensorAccessor<scalar_t, 2> A3
){
	// A3 = A1 + A2
	// M [L, C] ; mask [L, C]
	int L=A1.size(0), C=A1.size(1);

	for (int i=0; i<L; i++)
		for (int j=0; j<C; j++)
		{
			scalar_t s = A1[i][j] + A2[i][j];
			if (out_atomic_add)
				atomicAdd(&A3[i][j], s);
			else
				A3[i][j] = s;
		}
}


template <typename scalar_t>
__host__ __device__ inline void linear_attention_forward(
	at::TensorAccessor<scalar_t, 2> Q,
	at::TensorAccessor<scalar_t, 2> K,
	at::TensorAccessor<scalar_t, 2> V,
	at::TensorAccessor<scalar_t, 2> M,
	at::TensorAccessor<scalar_t, 2> O
){
	// 线性注意力-前向传播函数
	// Q [] ; K [] ; V [] ; M [] ; O [] ;
	scalar_t A_[Q_BLOCK_SIZE][KV_BLOCK_SIZE] = {0};
	int64_t A_sizes[2] = {Q_BLOCK_SIZE, KV_BLOCK_SIZE};
	int64_t A_strides[2] = {KV_BLOCK_SIZE, 1};

	at::TensorAccessor<scalar_t, 2> A((scalar_t*)A_, (int64_t*)A_sizes, (int64_t*)A_strides);

	// A = Q @ K.transpose(-1, -2)
	my_matmul<scalar_t, false, true, false>(Q, K, A);
	// A 在后面没用了，这里让Am复用A的空间
	auto &Am = A;
	// Am = A * M
	apply_mul_mask(Am, M);
	// 这里的 O 是大O的子集，会有多个线程同时相加和写入，所以这里O需要使用原子加法
	// O = Am @ V
	my_matmul<scalar_t, false, false, true>(A, V, O);
}


template <typename scalar_t>
__host__ __device__ inline void linear_attention_backward(
	// in
	at::TensorAccessor<scalar_t, 2> Q,
	at::TensorAccessor<scalar_t, 2> K,
	at::TensorAccessor<scalar_t, 2> V,
	at::TensorAccessor<scalar_t, 2> M,
	at::TensorAccessor<scalar_t, 2> dO,
	// out
	at::TensorAccessor<scalar_t, 2> dQ,
	at::TensorAccessor<scalar_t, 2> dK,
	at::TensorAccessor<scalar_t, 2> dV,
	at::TensorAccessor<scalar_t, 2> dM
){
	// 线性注意力-反向传播函数
	// Q [] ; K [] ; V [] ; M [] ; O [] ;

	// A[0] A ; A[1] dAm ; A[2] Am;
	scalar_t A_[3][Q_BLOCK_SIZE][KV_BLOCK_SIZE] = {0};
	int64_t A_sizes[3] = {3, Q_BLOCK_SIZE, KV_BLOCK_SIZE};
	int64_t A_strides[3] = {Q_BLOCK_SIZE * KV_BLOCK_SIZE, KV_BLOCK_SIZE, 1};
	at::TensorAccessor<scalar_t, 3> A_cache((scalar_t*)A_, (int64_t*)A_sizes, (int64_t*)A_strides);

	auto A = A_cache[0];
	auto dAm = A_cache[1];
	auto dA = A_cache[2];

	// A = Q @ K.transpose(-1, -2)
	my_matmul<scalar_t, false, true, false>(Q, K, A);

	// dAm = dO @ V.transpose(-1, -2)
	my_matmul<scalar_t, false, true, false>(dO, V, dAm);
	// dM = dAm * A ; 注意这里的dM是子集
	mul_arr<scalar_t, true>(dAm, A, dM);
	// dA = dAm * M
	mul_arr<scalar_t, false>(dAm, M, dA);

	// A后面没用了，重用 A ，现在 A 是 Am
	// Am = A * M
	auto& Am = A;
	apply_mul_mask(Am, M);
	// dV = Am.transpose(-1, -2) @ dO ; 注意这里的dV是子集
	my_matmul<scalar_t, true, false, true>(Am, dO, dV);
	// dK = dA.transpose(-1, -2) @ Q ; 注意这里的dK是子集
	my_matmul<scalar_t, true, false, true>(dA, Q, dK);
	// dQ = dA @ K ; 注意这里的dQ是子集
	my_matmul<scalar_t, false, false, true>(dA, K, dQ);
}


template<typename scalar_t>
__global__ void linear_attention_forward_kernel(
	at::PackedTensorAccessor<scalar_t, 5> Q,
	at::PackedTensorAccessor<scalar_t, 5> K,
	at::PackedTensorAccessor<scalar_t, 5> V,
	at::PackedTensorAccessor<scalar_t, 6> M,
	at::PackedTensorAccessor<scalar_t, 5> O,
	int64_t B,
	int64_t H,
	int64_t qL,
	int64_t kL,
	int64_t qN,
	int64_t kN,
	int64_t C
){
	// 获得当前线程分块信息
	int64_t cur_Bi = blockIdx.x / H;
	int64_t cur_Hi = blockIdx.x % H;
	int64_t cur_Qi = blockIdx.y;
	int64_t cur_Ki = threadIdx.x;

	// 获得当前线程的分块数据
	// printf("B%d H%d Q%d K%d\n", cur_Bi, cur_Hi, cur_Qi, cur_Ki);
	at::TensorAccessor<scalar_t, 2> block_q = Q[cur_Bi][cur_Hi][cur_Qi];
	at::TensorAccessor<scalar_t, 2> block_k = K[cur_Bi][cur_Hi][cur_Ki];
	at::TensorAccessor<scalar_t, 2> block_v = V[cur_Bi][cur_Hi][cur_Ki];
	at::TensorAccessor<scalar_t, 2> block_m = M[cur_Bi][cur_Hi][cur_Qi][cur_Ki];
	at::TensorAccessor<scalar_t, 2> block_o = O[cur_Bi][cur_Hi][cur_Qi];

	linear_attention_forward(block_q, block_k, block_v, block_m, block_o);
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
	int64_t B,
	int64_t H,
	int64_t qL,
	int64_t kL,
	int64_t qN,
	int64_t kN,
	int64_t C
){
	// 获得当前线程分块信息
	int64_t cur_Bi = blockIdx.x / H;
	int64_t cur_Hi = blockIdx.x % H;
	int64_t cur_Qi = blockIdx.y;
	int64_t cur_Ki = threadIdx.x;

	// 获得当前线程的分块数据
	// printf("B%d H%d Q%d K%d\n", cur_Bi, cur_Hi, cur_Qi, cur_Ki);
	at::TensorAccessor<scalar_t, 2> block_q = Q[cur_Bi][cur_Hi][cur_Qi];
	at::TensorAccessor<scalar_t, 2> block_k = K[cur_Bi][cur_Hi][cur_Ki];
	at::TensorAccessor<scalar_t, 2> block_v = V[cur_Bi][cur_Hi][cur_Ki];
	at::TensorAccessor<scalar_t, 2> block_m = M[cur_Bi][cur_Hi][cur_Qi][cur_Ki];
	at::TensorAccessor<scalar_t, 2> block_do = dO[cur_Bi][cur_Hi][cur_Qi];
	//
	at::TensorAccessor<scalar_t, 2> block_dq = dQ[cur_Bi][cur_Hi][cur_Qi];
	at::TensorAccessor<scalar_t, 2> block_dk = dK[cur_Bi][cur_Hi][cur_Ki];
	at::TensorAccessor<scalar_t, 2> block_dv = dV[cur_Bi][cur_Hi][cur_Ki];
	at::TensorAccessor<scalar_t, 2> block_dm = dM[cur_Bi][cur_Hi][cur_Qi][cur_Ki];

	linear_attention_backward(
		block_q, block_k, block_v, block_m, block_do,
		block_dq, block_dk, block_dv, block_dm
	);
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
	int B = blockIdx.x;
	
	printf("g%d ", B);
	linear_attention_forward(Q[B], K[B], V[B], M[B], O[B]);
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
	int B = blockIdx.x;
	
	printf("g%d ", B);
	linear_attention_backward(
		Q[B], K[B], V[B], M[B], dO[B],
		dQ[B], dK[B], dV[B], dM[B]
	);
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
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	using scalar_t = float;

	// 在外面使用 expand 支持 mask 的 B == 1 和 H == 1 ，里面不写支持
	// 在外面使用 expand 支持 k_head == 1 ，里面不写支持

	int64_t B=Q.size(0), H=Q.size(1), qL=Q.size(2), C=Q.size(3);
	int64_t kL=K.size(2);

	O.zero_();

	TORCH_CHECK(qL % Q_BLOCK_SIZE == 0, "Bad");
	TORCH_CHECK(kL % KV_BLOCK_SIZE == 0, "Bad");

	int64_t qN = qL / Q_BLOCK_SIZE;
	int64_t kN = kL / KV_BLOCK_SIZE;

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

	int64_t B=Q.size(0), H=Q.size(1), qL=Q.size(2), C=Q.size(3);
	int64_t kL=K.size(2);

	dQ.zero_();
	dK.zero_();
	dV.zero_();
	dM.zero_();

	TORCH_CHECK(qL % Q_BLOCK_SIZE == 0, "Bad qL");
	TORCH_CHECK(kL % KV_BLOCK_SIZE == 0, "Bad kL");
	TORCH_CHECK(kL % KV_BLOCK_SIZE == 0, "Bad");

	int64_t qN = qL / Q_BLOCK_SIZE;
	int64_t kN = kL / KV_BLOCK_SIZE;

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

	dim3 grid(B*H, qN);
	dim3 block(kN);

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
