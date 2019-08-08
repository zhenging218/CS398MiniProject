#include "precomp.cuh"
#include "bitboard.h"
#include "gpuminimax.h"

namespace Checkers
{
	namespace GPUMinimax
	{
		__global__ void master_white_next_kernel(int *placement, int X, GPUBitBoard const *boards, int num_boards, int depth, int turns)
		{
			int tx = threadIdx.x;
			__shared__ Minimax::utility_type v[32];
			__shared__ Minimax::utility_type *ret_v;
			int t_placement;

			if (tx == 0)
			{
				t_placement = *placement;
				cudaMalloc(&ret_v, sizeof(Minimax::utility_type) * num_boards);
			}

			__syncthreads();

			if (tx < num_boards)
			{
				white_min_kernel << <dim3(1, 1, 1), dim3(32, 1, 1) >> > (ret_v + tx, boards[tx], -Infinity, Infinity, depth, turns);
				v[tx] = ret_v[tx];
			}

			__syncthreads();

			if (tx == 0)
			{
				cudaFree(ret_v);
				for (int i = 0; i < num_boards; ++i)
				{
					if (X < v[i])
					{
						X = v[i];
						t_placement = i;
					}
				}

				*placement = t_placement;
			}

			__syncthreads();
		}

		__global__ void master_black_next_kernel(int *placement, int X, GPUBitBoard const *boards, int num_boards, int depth, int turns)
		{
			int tx = threadIdx.x;
			__shared__ Minimax::utility_type v[32];
			__shared__ Minimax::utility_type *ret_v;
			int t_placement;

			if (tx == 0)
			{
				t_placement = *placement;
				cudaMalloc(&ret_v, sizeof(Minimax::utility_type) * num_boards);

			}

			__syncthreads();

			if (tx < num_boards)
			{
				black_min_kernel << <dim3(1, 1, 1), dim3(32, 1, 1) >> > (ret_v + tx, boards[tx], -Infinity, Infinity, depth, turns);
				v[tx] = ret_v[tx];
			}

			__syncthreads();

			if (tx == 0)
			{
				cudaFree(ret_v);
				for (int i = 0; i < num_boards; ++i)
				{
					if (X < v[i])
					{
						X = v[i];
						t_placement = i;
					}
				}

				*placement = t_placement;
			}

			__syncthreads();
		}
	}
}