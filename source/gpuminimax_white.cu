#include "precomp.cuh"
#include "bitboard.h"
#include "gpuminimax.h"

#include <helper_cuda.h>
#include <cuda_runtime.h>

namespace Checkers
{
	// all master kernels should already have v and the utility array values initialised to the first v value computed by the CPU PV-Split.

	namespace GPUMinimax
	{
		__global__ void white_min_kernel(Minimax::utility_type *v, GPUBitBoard src, int alpha, int beta, int depth, int turns)
		{
			int tx = threadIdx.x;
			int t_beta = beta;
			int t_v = Minimax::Infinity;
			__shared__ bool terminated;
			__shared__ Minimax::utility_type utilities[32];
			GPUBitBoard new_boards[32];

			if (tx == 0)
			{
				Minimax::utility_type terminal_value = 0;
				if (src.valid)
				{
					terminated = GetBlackUtility(src, terminal_value, depth, turns);
					if (terminated)
						*v = terminal_value;

				}
				else
					terminated = true;

			}

			__syncthreads();

			if (terminated)
			{
				return;
			}
			else
			{
				if (tx < 32)
				{
					utilities[tx] = Minimax::Infinity;
				}
				__syncthreads();

				Minimax::utility_type * utility;
				cudaMalloc(&utility, sizeof(Minimax::utility_type));
				*utility = utilities[tx];

				// in the max kernel, use gen_black_move_type instead
				int gen_black_move_type = (int)(GPUBitBoard::GetBlackJumps(src) != 0);
				GPUBitBoard *end = new_boards;
				gen_black_move[gen_black_move_type](1u << tx, end, src);
				int frontier_size = end - new_boards;

				GPUBitBoard *frontier;
				cudaMalloc(&frontier, sizeof(GPUBitBoard) * frontier_size);
				memcpy(frontier, new_boards, sizeof(GPUBitBoard) * frontier_size);

				master_white_max_kernel << <dim3(1, 1, 1), dim3(32, 1, 1) >> > (utility, frontier, frontier_size, alpha, t_beta, depth - 1, turns - 1);
				utilities[tx] = *utility;

				cudaFree(frontier);
				cudaFree(utility);

				__syncthreads();

				if (tx == 0)
				{
					// final ab-pruning for this node
					for (int i = 0; i < 32; ++i)
					{
						t_v = min(utilities[i], t_v);
						if (t_v < alpha)
						{
							break;
						}
						else
						{
							beta = min(utilities[i], beta);
						}
					}
					*v = t_v;
				}

				__syncthreads();
			}
		}

		__global__ void white_max_kernel(Minimax::utility_type *v, GPUBitBoard src, int alpha, int beta, int depth, int turns)
		{
			int tx = threadIdx.x;
			int t_alpha = alpha;
			int t_v = -Minimax::Infinity;
			__shared__ bool terminated;
			__shared__ Minimax::utility_type utilities[32];
			GPUBitBoard new_boards[32];

			if (!tx)
			{
				Minimax::utility_type terminal_value = 0;
				if (src.valid)
				{
					terminated = GetWhiteUtility(src, terminal_value, depth, turns);
					if (terminated)
						*v = terminal_value;

				}
				else
					terminated = true;

			}

			__syncthreads();

			if (terminated)
			{
				return;
			}
			else
			{
				if (tx < 32)
				{
					utilities[tx] = -Minimax::Infinity;
				}

				__syncthreads();

				Minimax::utility_type * utility;
				cudaMalloc(&utility, sizeof(Minimax::utility_type));
				*utility = utilities[tx];
				GPUBitBoard *end = new_boards;
				// in the max kernel, use gen_black_move_type instead
				int gen_white_move_type = (int)(GPUBitBoard::GetWhiteJumps(src) != 0);
				gen_white_move[gen_white_move_type](1u << tx, end, src);

				int frontier_size = end - new_boards;

				GPUBitBoard *frontier;
				cudaMalloc(&frontier, sizeof(GPUBitBoard) * frontier_size);
				memcpy(frontier, new_boards, sizeof(GPUBitBoard) * frontier_size);

				master_white_min_kernel << <dim3(1, 1, 1), dim3(32, 1, 1) >> > (utility, frontier, frontier_size, t_alpha, beta, depth - 1, turns - 1);
				utilities[tx] = *utility;

				cudaFree(frontier);
				cudaFree(utility);

				__syncthreads();

				if (tx == 0)
				{
					// final ab-pruning for this node
					for (int i = 0; i < 32; ++i)
					{
						t_v = max(utilities[i], t_v);
						if (t_v > beta)
						{
							break;
						}
						else
						{
							alpha = max(utilities[i], alpha);
						}
					}

					*v = t_v;
				}

				__syncthreads();
			}
		}

		
	}
}