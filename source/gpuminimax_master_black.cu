#include "precomp.cuh"
#include "bitboard.h"
#include "gpuminimax.h"

namespace Checkers
{
	namespace GPUMinimax
	{
		__global__ void master_black_max_kernel(Minimax::utility_type *v, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns)
		{
			int tx = threadIdx.x;
			Minimax::utility_type t_v = *v;
			__shared__ Minimax::utility_type utility[32];
			if (tx < num_boards)
			{
				utility[tx] = black_min_device(src[tx], depth - 1, turns - 1, alpha, beta);
			}

			__syncthreads();

			if (tx == 0)
			{
				for (int i = 0; i < num_boards; ++i)
				{
					t_v = max(utility[tx], t_v);
					if (t_v > beta)
						break;
					alpha = max(alpha, t_v);
				}
				*v = t_v;
			}

			__syncthreads();
		}

		__global__ void master_black_min_kernel(Minimax::utility_type *v, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns)
		{
			int tx = threadIdx.x;
			Minimax::utility_type t_v = *v;
			__shared__ Minimax::utility_type *utility;
			__shared__ Minimax::utility_type t_utility[32];

			if (tx == 0)
			{
				utility[tx] = black_max_device(src[tx], depth - 1, turns - 1, alpha, beta);
			}

			__syncthreads();

			if (tx == 0)
			{
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