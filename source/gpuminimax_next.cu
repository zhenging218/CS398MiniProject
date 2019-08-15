#include "precomp.cuh"
#include "bitboard.h"
#include "gpuminimax.h"

namespace Checkers
{
	namespace GPUMinimax
	{
		__global__ void white_next_kernel(int *placement, utility_type *v, utility_type X, GPUBitBoard const *boards, int num_boards, int depth, int turns)
		{
			int tx = threadIdx.x;
			int bx = blockIdx.x;

			__shared__ int frontier_size;
			__shared__ GPUBitBoard frontier[32];
			__shared__ utility_type t_v[32];
			__shared__ utility_type alpha;
			__shared__ utility_type beta;
			__shared__ int gen_board_type;
			__shared__ bool terminated;

			if (tx == 0)
			{
				frontier_size = 0;
				alpha = -Infinity;
				beta = Infinity;
				utility_type terminal_value = 0;
				if (terminated = GetWhiteUtility(boards[bx], terminal_value, depth, turns))
				{
					v[bx] = terminal_value;
				}
				else
				{
					gen_board_type = (GPUBitBoard::GetBlackJumps(boards[bx]) != 0) ? 1 : 0;
				}
			}

			__syncthreads();

			if (!terminated)
			{
				gen_black_move_atomic[gen_board_type](1u << tx, boards[bx], frontier, &frontier_size);

				__syncthreads();

				if (tx < frontier_size)
				{
					t_v[tx] = explore_white_frontier(frontier[tx], alpha, beta, NodeType::MAX, depth - 1, turns - 1);
				}

				__syncthreads();

				// min
				if (tx == 0)
				{
					utility_type t_x = Infinity;
					for(int i = 0; i < frontier_size; ++i)
					{
						t_x = MIN(t_v[i], t_x);
						if (t_x < alpha)
						{
							break;
						}
						beta = MIN(beta, t_x);
					}

					v[bx] = t_x;
				}
			}

			__syncthreads();

			// max
			if (bx == 0 && tx == 0)
			{
				int t_placement = *placement;
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

		__global__ void black_next_kernel(int *placement, utility_type *v, utility_type X, GPUBitBoard const *boards, int num_boards, int depth, int turns)
		{
			int tx = threadIdx.x;
			int bx = blockIdx.x;

			__shared__ int frontier_size;
			__shared__ GPUBitBoard frontier[32];
			__shared__ utility_type t_v[32];
			__shared__ utility_type alpha;
			__shared__ utility_type beta;
			__shared__ int gen_board_type;
			__shared__ bool terminated;

			if (tx == 0)
			{
				frontier_size = 0;
				alpha = -Infinity;
				beta = Infinity;
				utility_type terminal_value = 0;
				if (terminated = GetBlackUtility(boards[bx], terminal_value, depth, turns))
				{
					v[bx] = terminal_value;
				}
				else
				{
					gen_board_type = (GPUBitBoard::GetWhiteJumps(boards[bx]) != 0) ? 1 : 0;
				}

			}

			__syncthreads();

			if (!terminated)
			{
				gen_white_move_atomic[gen_board_type](1u << tx, boards[bx], frontier, &frontier_size);   

				__syncthreads();

				if (tx < frontier_size)
				{
					t_v[tx] = explore_black_frontier(frontier[tx], alpha, beta, NodeType::MAX, depth - 1, turns - 1);
				}

				__syncthreads();


				// min
				if (tx == 0)
				{
					utility_type t_x = Infinity;
					for (int i = 0; i < frontier_size; ++i)
					{
						t_x = MIN(t_v[i], t_x);
						if (t_x < alpha)
						{
							break;
						}
						beta = MIN(beta, t_x);
					}

					v[bx] = t_x;
				}
			}

			__syncthreads();

			// max
			if (bx == 0 && tx == 0)
			{
				int t_placement = *placement;
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