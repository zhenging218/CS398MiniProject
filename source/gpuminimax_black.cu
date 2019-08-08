#include "precomp.cuh"
#include "bitboard.h"
#include "gpuminimax.h"

namespace Checkers
{
	namespace GPUMinimax
	{
		__global__ void black_min_kernel(Minimax::utility_type *v, GPUBitBoard src, int alpha, int beta, int depth, int turns)
		{
			int tx = threadIdx.x;
			int t_beta = beta;
			int t_v = Minimax::Infinity;
			__shared__ bool terminated;
			__shared__ Minimax::utility_type utilities[32];
			__shared__ cudaStream_t streams[4];
			__shared__ bool valid[32];
			cudaEvent_t stream_events[32];
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

					if (tx == 0)
					{
						cudaStreamCreateWithFlags(&streams[0], cudaStreamNonBlocking);
						cudaStreamCreateWithFlags(&streams[1], cudaStreamNonBlocking);
						cudaStreamCreateWithFlags(&streams[2], cudaStreamNonBlocking);
						cudaStreamCreateWithFlags(&streams[3], cudaStreamNonBlocking);
					}
				}
				__syncthreads();

				// in the max kernel, use gen_black_move_type instead
				int gen_white_move_type = (int)(GPUBitBoard::GetWhiteJumps(src) != 0);
				GPUBitBoard *end = new_boards;
				gen_white_move[gen_white_move_type](1u << tx, end, src);
				int frontier_size = end - new_boards;
				valid[tx] = (frontier_size != 0);
				Minimax::utility_type * utility;
				cudaMalloc(&utility, sizeof(Minimax::utility_type) * frontier_size);

				for (int i = 0; i < frontier_size; ++i)
				{
					black_max_kernel << <dim3(1, 1, 1), dim3(32, 1, 1), 0, streams[i % 4] >> >(utility + i, new_boards[i], alpha, t_beta, depth - 1, turns - 1);
					cudaStreamWaitEvent(streams[i % 4], stream_events[i], 0);
				}

				for (int i = 0; i < frontier_size; ++i)
				{
					utilities[tx] = min(utility[i], utilities[tx]);
					if (utilities[tx] < alpha)
					{
						break;
					}
					else
					{
						t_beta = min(utilities[tx], t_beta);
					}
				}

				cudaFree(utility);

				__syncthreads();

				if (tx == 0)
				{
					// all streams in the block should have completed processing by now.
					for (int i = 0; i < 4; ++i)
					{
						// cudaStreamSynchronize(streams[i]);
						cudaStreamDestroy(streams[i]);
					}

					// final ab-pruning for this node
					for (int i = 0; i < 32; ++i)
					{
						if (valid[i])
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
					}

					*v = t_v;
				}

				__syncthreads();
			}
		}

		__global__ void black_max_kernel(Minimax::utility_type *v, GPUBitBoard src, int alpha, int beta, int depth, int turns)
		{
			int tx = threadIdx.x;
			int t_alpha = alpha;
			int t_v = -Minimax::Infinity;
			__shared__ bool terminated;
			__shared__ Minimax::utility_type utilities[32];
			__shared__ cudaStream_t streams[4];
			__shared__ bool valid[32];
			cudaEvent_t stream_events[32];
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
					utilities[tx] = -Minimax::Infinity;

					if (tx == 0)
					{
						cudaStreamCreateWithFlags(&streams[0], cudaStreamNonBlocking);
						cudaStreamCreateWithFlags(&streams[1], cudaStreamNonBlocking);
						cudaStreamCreateWithFlags(&streams[2], cudaStreamNonBlocking);
						cudaStreamCreateWithFlags(&streams[3], cudaStreamNonBlocking);
					}
				}
				__syncthreads();

				// in the max kernel, use gen_black_move_type instead
				int gen_black_move_type = (int)(GPUBitBoard::GetBlackJumps(src) != 0);
				GPUBitBoard *end = new_boards;
				gen_black_move[gen_black_move_type](1u << tx, end, src);

				int frontier_size = end - new_boards;
				valid[tx] = (frontier_size != 0);
				Minimax::utility_type * utility;
				cudaMalloc(&utility, sizeof(Minimax::utility_type) * frontier_size);

				for (int i = 0; i < frontier_size; ++i)
				{
					black_min_kernel << <dim3(1, 1, 1), dim3(32, 1, 1), 0, streams[i % 4] >> >(utility + i, new_boards[i], t_alpha, beta, depth - 1, turns - 1);
					cudaStreamWaitEvent(streams[i % 4], stream_events[i], 0);
				}

				for (int i = 0; i < frontier_size; ++i)
				{
					utilities[tx] = max(utility[i], utilities[tx]);
					if (utilities[tx] > beta)
					{
						break;
					}
					else
					{
						t_alpha = max(utilities[tx], t_alpha);
					}
				}

				cudaFree(utility);

				__syncthreads();

				if (tx == 0)
				{
					// all streams in the block should have completed processing by now.
					for (int i = 0; i < 4; ++i)
					{
						cudaStreamDestroy(streams[i]);
					}

					// final ab-pruning for this node
					for (int i = 0; i < 32; ++i)
					{
						if (valid[i])
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
					}

					*v = t_v;
				}

				__syncthreads();
			}
		}
	}
}