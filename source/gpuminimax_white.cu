#include "precomp.cuh"
#include "bitboard.h"
#include "gpuminimax.h"

namespace Checkers
{
	namespace GPUMinimax
	{
		__device__ utility_type explore_white_frontier(GPUBitBoard board, utility_type alpha, utility_type beta, NodeType node_type, int depth, int turns)
		{
			GPUBitBoard frontier[32];
			int frontier_size = 0;
			int v = (node_type == NodeType::MAX) ? -Infinity : Infinity;

			int gen_board_type;

			utility_type terminal_value = 0;
			if (GetWhiteUtility(board, terminal_value, depth, turns))
			{
				return terminal_value;
			}
			if (node_type == NodeType::MAX)
			{
				gen_board_type = (GPUBitBoard::GetWhiteJumps(board) != 0) ? 1 : 0;
			}
			else
			{
				gen_board_type = (GPUBitBoard::GetBlackJumps(board) != 0) ? 1 : 0;
			}

			if (node_type == NodeType::MAX)
			{
				// if dynamic parallelism is possible, can call another kernel here
				for (int i = 0; i < 32; ++i)
				{
					gen_white_move[gen_board_type](1u << i, board, frontier, frontier_size);
				}

				for (int j = 0; j < frontier_size; ++j)
				{
					v = GET_MAX(explore_white_frontier(frontier[j], alpha, beta, node_type + 1, depth - 1, turns - 1), v);
					if (v > beta)
					{
						break;
					}
					alpha = GET_MAX(alpha, v);
				}
			}
			else
			{
				// if dynamic parallelism is possible, can call another kernel here
				for (int i = 0; i < 32; ++i)
				{
					gen_black_move[gen_board_type](1u << i, board, frontier, frontier_size);
				}

				for (int j = 0; j < frontier_size; ++j)
				{
					v = GET_MIN(explore_white_frontier(frontier[j], alpha, beta, node_type + 1, depth - 1, turns - 1), v);
					if (v < alpha)
					{
						break;
					}
					beta = GET_MIN(beta, v);
				}
			}
			return v;
		}

		__global__ void white_kernel(utility_type *v, GPUBitBoard const *boards, int num_boards, utility_type alpha, utility_type beta, NodeType node_type, int depth, int turns)
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
				if (terminated = GetWhiteUtility(boards[bx], terminal_value, depth, turns))
				{
					v[bx] = terminal_value;
				}
				else
				{
					if ((node_type + 1) == NodeType::MAX)
					{
						gen_board_type = (GPUBitBoard::GetWhiteJumps(boards[bx]) != 0) ? 1 : 0;
					}
					else
					{
						gen_board_type = (GPUBitBoard::GetBlackJumps(boards[bx]) != 0) ? 1 : 0;
					}
				}
			}
			__syncthreads();

			if (!terminated)
			{
				if ((node_type + 1) == NodeType::MAX)
				{
					gen_white_move_atomic[gen_board_type](1u << tx, boards[bx], frontier, &frontier_size);
				}
				else
				{
					gen_black_move_atomic[gen_board_type](1u << tx, boards[bx], frontier, &frontier_size);
				}

				__syncthreads();

				if (tx < frontier_size)
				{
					t_v[tx] = explore_white_frontier(frontier[tx], alpha, beta, node_type + 2, depth - 1, turns - 1);
				}
				else
				{
					t_v[tx] = node_type == NodeType::MAX ? Infinity : -Infinity;
				}

				__syncthreads();

				if ((node_type + 1) == NodeType::MAX)
				{
					for (int i = 1; i < 32; i *= 2)
					{
						if (tx + i < 32)
						{
							t_v[tx] = GET_MAX(t_v[tx], t_v[tx + i]);
						}
					}
				}
				else
				{
					for (int i = 1; i < 32; i *= 2)
					{
						if (tx + i < 32)
						{
							t_v[tx] = GET_MIN(t_v[tx], t_v[tx + i]);
						}
					}
				}

				__syncthreads();

				if (tx == 0)
				{
					v[bx] = t_v[0];
				}
			}

			__syncthreads();

			if (bx == 0)
			{
				if (tx < num_boards)
				{
					t_v[tx] = v[tx];
				}
				else
				{
					t_v[tx] = node_type == NodeType::MAX ? Infinity : -Infinity;
				}

				__syncthreads();

				if ((node_type) == NodeType::MAX)
				{
					for (int i = 1; i < 32; i *= 2)
					{
						if (tx + i < 32)
						{
							t_v[tx] = GET_MAX(t_v[tx], t_v[tx + i]);
						}
					}
				}
				else
				{
					for (int i = 1; i < 32; i *= 2)
					{
						if (tx + i < 32)
						{
							t_v[tx] = GET_MIN(t_v[tx], t_v[tx + i]);
						}
					}
				}

				__syncthreads();

				if (tx < num_boards)
				{
					v[tx] = t_v[tx];
				}
			}

			__syncthreads();
		}
	}
}