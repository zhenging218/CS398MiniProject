#include "precomp.cuh"
#include "bitboard.h"
#include "gpuminimax.h"

namespace Checkers
{
	namespace GPUMinimax
	{
		__device__ utility_type explore_black_frontier(GPUBitBoard board, utility_type alpha, utility_type beta, NodeType node_type, int depth, int turns)
		{
			GPUBitBoard frontier[32];
			int frontier_size = 0;
			int v = (node_type == NodeType::MAX) ? -Infinity : Infinity;

			int gen_board_type;

			utility_type terminal_value = 0;
			if (GetBlackUtility(board, terminal_value, depth, turns))
			{
				return terminal_value;
			}

			if (node_type == NodeType::MAX)
			{
				gen_board_type = (GPUBitBoard::GetBlackJumps(board) != 0) ? 1 : 0;
			}
			else
			{
				gen_board_type = (GPUBitBoard::GetWhiteJumps(board) != 0) ? 1 : 0;
			}

			if (node_type == NodeType::MAX)
			{
				// if dynamic parallelism is possible, can call another kernel here
				for (int i = 0; i < 32; ++i)
				{
					gen_black_move[gen_board_type](1u << i, board, frontier, frontier_size);
				}

				while (frontier_size > 0)
				{
					v = max(explore_black_frontier(frontier[--frontier_size], alpha, beta, node_type + 1, depth - 1, turns - 1), v);
					if (v > beta)
					{
						break;
					}
					alpha = max(alpha, v);
					
				}
			}
			else
			{
				// if dynamic parallelism is possible, can call another kernel here
				for (int i = 0; i < 32; ++i)
				{
					gen_white_move[gen_board_type](1u << i, board, frontier, frontier_size);
				}

				while (frontier_size > 0)
				{
					v = min(explore_black_frontier(frontier[--frontier_size], alpha, beta, node_type + 1, depth - 1, turns - 1), v);
					
					if (v < alpha)
					{
						break;
					}
					beta = min(beta, v);
				}
			}

			return v;
		}

		__global__ void black_kernel(utility_type *v, GPUBitBoard const *boards, int num_boards, utility_type alpha, utility_type beta, NodeType node_type, int depth, int turns)
		{
			int tx = threadIdx.x;
			int bx = blockIdx.x;

			__shared__ int frontier_size;
			__shared__ int gen_board_type;
			__shared__ GPUBitBoard frontier[32];
			__shared__ utility_type t_v[32];
			__shared__ bool terminated;

			if (tx == 0)
			{
				frontier_size = 0;
				utility_type terminal_value = 0;
				if (terminated = GetBlackUtility(boards[bx], terminal_value, depth, turns))
				{
					v[bx] = terminal_value;
				}
				else
				{
					if ((node_type + 1) == NodeType::MAX)
					{
						gen_board_type = (GPUBitBoard::GetBlackJumps(boards[bx]) != 0) ? 1 : 0;
					}
					else
					{
						gen_board_type = (GPUBitBoard::GetWhiteJumps(boards[bx]) != 0) ? 1 : 0;
					}
				}
			}
			__syncthreads();

			if (!terminated)
			{
				if ((node_type + 1) == NodeType::MAX)
				{
					gen_black_move_atomic[gen_board_type](1u << tx, boards[bx], frontier, &frontier_size);
				}
				else
				{
					gen_white_move_atomic[gen_board_type](1u << tx, boards[bx], frontier, &frontier_size);
				}

				__syncthreads();

				if (tx < frontier_size)
				{
					t_v[tx] = explore_black_frontier(frontier[tx], alpha, beta, node_type + 2, depth - 1, turns - 1);
				}

				__syncthreads();

				if (tx == 0)
				{
					utility_type t_x;
					// ab-prune t_v and send the last value to v[bx].
					if ((node_type + 1) == NodeType::MAX)
					{
						t_x = -Infinity;
						while (frontier_size > 0)
						{
							t_x = max(t_v[--frontier_size], t_x);
	
							if (t_x > beta)
							{
								break;
							}
							alpha = max(alpha, t_x);
							
						}
					}
					else
					{
						t_x = Infinity;
						while (frontier_size > 0)
						{
							t_x = min(t_v[--frontier_size], t_x);
						
							if (t_x < alpha)
							{
								break;
							}
							beta = min(beta, t_x);
						}
					}

					v[bx] = t_x;
				}
			}

			__syncthreads();

			if (bx == 0 && tx == 0)
			{
				utility_type X;
				// ab-prune v and send the last value to v[0].
				if (node_type == NodeType::MAX)
				{
					X = -Infinity;
					for (int i = 0; i < num_boards; ++i)
					{
						X = max(v[i], X);
						
						if (X > beta)
						{
							break;
						}
						alpha = max(alpha, X);
				
					}
				}
				else
				{
					X = Infinity;
					for (int i = 0; i < num_boards; ++i)
					{
						X = min(v[i], X);
						
						if (X < alpha)
						{
							break;
						}
						beta = min(beta, X);

					}
				}

				v[0] = X;
			}

			__syncthreads();
		}
	}
}