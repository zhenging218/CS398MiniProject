#include "precomp.cuh"
#include "bitboard.h"
#include "gpuminimax.h"

namespace Checkers
{
	namespace GPUMinimax
	{
		__global__ void master_white_next_kernel(int *placement, utility_type X, GPUBitBoard const *boards, int num_boards, int depth, int turns)
		{
			int tx = threadIdx.x;
			__shared__ Minimax::utility_type v[32];
			int t_placement = *placement;

			if (tx < num_boards)
			{
				v[tx] = white_min_device(boards[tx], depth, turns, -Infinity, Infinity);
				// cudaDeviceSynchronize();
			}
			__syncthreads();

			if (tx == 0)
			{
				for (int i = 0; i < num_boards; ++i)
				{
					if (X < v[i])
					{
						X = v[i];
						t_placement = i + 1;
					}
				}
				*placement = t_placement;
			}

			__syncthreads();
		}

		__global__ void master_black_next_kernel(int *placement, utility_type X, GPUBitBoard const *boards, int num_boards, int depth, int turns)
		{
			int tx = threadIdx.x;
			__shared__ Minimax::utility_type v[32];
			int t_placement = *placement;

			if (tx < num_boards)
			{
				v[tx] = black_min_device(boards[tx], -Infinity, Infinity,depth, turns );
			}

			__syncthreads();

			if (tx == 0)
			{
				for (int i = 0; i < num_boards; ++i)
				{
					if (X < v[i])
					{
						X = v[i];
						t_placement = i + 1;
					}
				}
				*placement = t_placement;
			}

			__syncthreads();
		}
	}
}