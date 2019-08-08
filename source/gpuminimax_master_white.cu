#include "precomp.cuh"
#include "bitboard.h"
#include "gpuminimax.h"

namespace Checkers
{
	namespace GPUMinimax
	{
		__global__ void master_white_max_kernel(Minimax::utility_type *v, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns)
		{
			int tx = threadIdx.x;
			Minimax::utility_type t_v = *v;
			__shared__ Minimax::utility_type *utility;
			__shared__ Minimax::utility_type t_utility[32];

			if (tx == 0)
			{
				cudaMalloc(&utility, sizeof(Minimax::utility_type) * num_boards);
				memset(utility, t_v, sizeof(Minimax::utility_type) * num_boards);
			}

			__syncthreads();

			if (tx < num_boards)
			{
				white_min_kernel << <dim3(1, 1, 1), dim3(32, 1, 1) >> > (utility + tx, src[tx], alpha, beta, depth - 1, turns - 1);
				t_utility[tx] = utility[tx];
			}

			__syncthreads();

			if (tx == 0)
			{
				cudaFree(utility);
				for (int i = 0; i < num_boards; ++i)
				{
					t_v = max(t_utility[tx], t_v);
					if (t_v > beta)
						break;
					alpha = max(alpha, t_v);
				}

				*v = t_v;
			}
			__syncthreads();
		}

		__global__ void master_white_min_kernel(Minimax::utility_type *v, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns)
		{
			int tx = threadIdx.x;
			Minimax::utility_type t_v = *v;
			__shared__ Minimax::utility_type *utility;
			__shared__ Minimax::utility_type t_utility[32];

			if (tx == 0)
			{
				cudaMalloc(&utility, sizeof(Minimax::utility_type) * num_boards);
				memset(utility, t_v, sizeof(Minimax::utility_type) * num_boards);
			}

			__syncthreads();

			if (tx < num_boards)
			{
				white_max_kernel << <dim3(1, 1, 1), dim3(32, 1, 1) >> > (utility + tx, src[tx], alpha, beta, depth - 1, turns - 1);
				t_utility[tx] = utility[tx];
			}

			__syncthreads();

			if (tx == 0)
			{
				cudaFree(utility);
				for (int i = 0; i < num_boards; ++i)
				{
					t_v = min(t_utility[tx], t_v);
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