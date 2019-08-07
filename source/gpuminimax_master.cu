#include "precomp.cuh"
#include "bitboard.h"
#include "gpuminimax.h"

#include <helper_cuda.h>

namespace Checkers
{
	namespace GPUMinimax
	{
		__global__ void master_white_max_kernel(utility_type *v, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns)
		{
			int tx = threadIdx.x;
			utility_type t_v = -Minimax::Infinity;
			__shared__ utility_type *utility;
			__shared__ utility_type t_utility[32];

			if (tx == 0)
			{
				cudaMalloc(&utility, sizeof(utility_type) * num_boards);
			}

			__syncthreads();

			if (tx < num_boards)
			{
				white_min_kernel << <dim3(1, 1, 1), dim3(32, 1, 1) >> > (utility + index, src[index], alpha, beta, depth - 1, turns - 1);
				t_utility = utility[tx];
			}

			__syncthreads();

			if (tx == 0)
			{
				cudaFree(utility);
				for (int i = 0; i < num_boards; ++i)
				{
					t_v = max(t_utility[index], t_v);
					if (t_v > beta)
						break;
					alpha = max(alpha, t_v);
				}

				*v = t_v;
			}
			__syncthreads();
		}

		__global__ void master_white_min_kernel(utility_type *v, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns)
		{
			int tx = threadIdx.x;
			utility_type t_v = Minimax::Infinity;
			__shared__ utility_type *utility;
			__shared__ utility_type t_utility[32];

			if (tx == 0)
			{
				cudaMalloc(&utility, sizeof(utility_type) * num_boards);
			}

			__syncthreads();

			if (tx < num_boards)
			{
				white_max_kernel << <dim3(1, 1, 1), dim3(32, 1, 1) >> > (utility + index, src[index], alpha, beta, depth - 1, turns - 1);
				t_utility[tx] = utility[tx];
			}
			__syncthreads();

			if (index == 0)
			{
				cudaFree(utility);
				for (int i = 0; i < num_boards; ++i)
				{
					t_v = min(t_utility[index], t_v);
					if (t_v < alpha)
						break;
					beta = min(beta, t_v);
				}
				*v = t_v;
			}
			__syncthreads();
		}

		__global__ void master_black_max_kernel(utility_type *v, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns)
		{
			int tx = threadIdx.x;
			utility_type t_v = -Minimax::Infinity;
			__shared__ utility_type *utility;
			__shared__ utility_type t_utility[32];

			if (tx == 0)
			{
				cudaMalloc(&utility, sizeof(utility_type) * num_boards);
			}

			__syncthreads();

			if (index < num_boards)
			{
				black_min_kernel << <dim3(1, 1, 1), dim3(32, 1, 1) >> > (utility + index, src[index], alpha, beta, depth - 1, turns - 1);
				t_utility[tx] = utility[tx];
			}
			__syncthreads();

			if (tx == 0)
			{
				cudaFree(utility);
				for (int i = 0; i < num_boards; ++i)
				{
					t_v = max(utility[index], t_v);
					if (t_v > beta)
						break;
					alpha = max(alpha, t_v);
				}
				*v = t_v;
			}
			__syncthreads();
		}

		__global__ void master_black_min_kernel(utility_type *v, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns)
		{
			int tx = threadIdx.x;
			utility_type t_v = Minimax::Infinity;
			__shared__ utility_type *utility;
			__shared__ utility_type t_utility[32];

			if (tx == 0)
			{
				cudaMalloc(&utility, sizeof(utility_type) * num_boards);
			}

			__syncthreads();

			if (index < num_boards)
			{
				black_max_kernel << <dim3(1, 1, 1), dim3(32, 1, 1) >> > (utility + index, src + index, alpha, beta, depth - 1, turns - 1);

			}
			__syncthreads();

			if (tx == 0)
			{
				cudaFree(utility);
				for (int i = 0; i < num_boards; ++i)
				{
					t_v = min(utility[index], t_v);
					if (t_v < alpha)
						break;
					beta = min(beta, t_v);
				}
				*v = t_v;
			}
			__syncthreads();
		}
	}
}