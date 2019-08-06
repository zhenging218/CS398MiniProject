#include "precomp.cuh"
#include "gpuminimax.h"

#include <helper_cuda.h>

namespace Checkers
{
	// all master kernels should already have v and the utility array values initialised to the first v value computed by the CPU PV-Split.

	namespace GPUMinimax
	{
		// kernels
		__global__ void master_white_max_kernel(utility_type *v, utility_type *utility, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns)
		{
			int tx = threadIdx.x;
			int t_v = -Minimax::Infinity;
			int index = blockIdx.x * blockDim.x + tx;

			if (index < num_boards)
			{
				white_min_kernel << <dim3(1, 1, 1), dim3(32, 1, 1) >> > (utility + index, src[index], alpha, beta, depth - 1, turns - 1);

			}
			__syncthreads();

			if (!index)
			{
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

		__global__ void master_white_min_kernel(utility_type *v, utility_type *utility, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns)
		{
			int tx = threadIdx.x;
			int t_v = Minimax::Infinity;
			int index = blockIdx.x * blockDim.x + tx;

			if (index < num_boards)
			{
				white_max_kernel << <dim3(1, 1, 1), dim3(32, 1, 1) >> > (utility + index, src[index], alpha, beta, depth - 1, turns - 1);

			}
			__syncthreads();

			if (index == 0)
			{
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

		__global__ void white_min_kernel(utility_type *v, GPUBitBoard src, int alpha, int beta, int depth, int turns)
		{
			int tx = threadIdx.x;
			int t_beta = beta;
			int t_v = Minimax::Infinity;
			__shared__ bool terminated;
			__shared__ utility_type utilities[32];

			if (tx == 0)
			{
				utility_type terminal_value = 0;
				if (src->valid)
				{
					terminated = GPUGetWhiteUtility(src, &terminal_value, depth, turns);
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

				utility_type * utility;
				cudaMalloc(&utility, sizeof(utility_type) * 4);
				GPUBitBoard *new_boards;
				cudaMalloc(&new_boards, sizeof(GPUBitBoard) * 4);

				cudaStream_t streams[4];
				cudaStreamCreateWithFlags(&streams[0], cudaStreamNonBlocking);
				cudaStreamCreateWithFlags(&streams[1], cudaStreamNonBlocking);
				cudaStreamCreateWithFlags(&streams[2], cudaStreamNonBlocking);
				cudaStreamCreateWithFlags(&streams[3], cudaStreamNonBlocking);

				// in the max kernel, use gen_black_move_type instead
				int gen_black_move_type = (int)GPUBitBoard::GetBlackJumps(src) != 0);
				gen_black_move[gen_black_move_type](1u << tx, new_board, src);

				white_max_kernel << <dim3(1, 1, 1), dim3(32, 1, 1), streams[0] >> >(utility, new_boards[0], alpha, t_beta, depth - 1, turns - 1);
				white_max_kernel << <dim3(1, 1, 1), dim3(32, 1, 1), streams[1] >> >(utility + 1, new_boards[1], alpha, t_beta, depth - 1, turns - 1);
				white_max_kernel << <dim3(1, 1, 1), dim3(32, 1, 1), streams[2] >> >(utility + 2, new_boards[2], alpha, t_beta, depth - 1, turns - 1);
				white_max_kernel << <dim3(1, 1, 1), dim3(32, 1, 1), streams[3] >> >(utility + 3, new_boards[3], alpha, t_beta, depth - 1, turns - 1);


				// sync streams here

				for (int i = 0; i < 4; ++i)
				{
					cudaStreamSynchronize(streams[i]);
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
				cudaFree(new_boards);

				cudaStreamDestroy(streams[0]);
				cudaStreamDestroy(streams[1]);
				cudaStreamDestroy(streams[2]);
				cudaStreamDestroy(streams[3]);

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

		__global__ void white_max_kernel(utility_type *v, GPUBitBoard src, int alpha, int beta, int depth, int turns)
		{
			int tx = threadIdx.x;
			int t_alpha = alpha;
			int t_v = -Minimax::Infinity;
			__shared__ bool terminated;
			__shared__ utility_type utilities[32];

			if (!tx)
			{
				utility_type terminal_value = 0;
				if (src->valid)
				{
					terminated = GPUGetWhiteUtility(src, &terminal_value, depth, turns);
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

				utility_type * utility;
				cudaMalloc(&utility, sizeof(utility_type) * 4);
				utility[0] = utility[1] = utility[2] = utility[3] = utilities[tx];
				GPUBitBoard *new_boards;
				cudaMalloc(&new_boards, sizeof(GPUBitBoard) * 4);

				cudaStream_t streams[4];
				cudaStreamCreateWithFlags(&streams[0], cudaStreamNonBlocking);
				cudaStreamCreateWithFlags(&streams[1], cudaStreamNonBlocking);
				cudaStreamCreateWithFlags(&streams[2], cudaStreamNonBlocking);
				cudaStreamCreateWithFlags(&streams[3], cudaStreamNonBlocking);

				// in the max kernel, use gen_black_move_type instead
				int gen_white_move_type = (int)GPUBitBoard::GetWhiteJumps(src) != 0);
				gen_white_move[gen_white_move_type](1u << tx, new_board, src);

				white_min_kernel << <dim3(1, 1, 1), dim3(32, 1, 1), streams[0] >> >(utility, new_boards[0], t_alpha, beta, depth - 1, turns - 1);
				white_min_kernel << <dim3(1, 1, 1), dim3(32, 1, 1), streams[1] >> >(utility + 1, new_boards[1], t_alpha, beta, depth - 1, turns - 1);
				white_min_kernel << <dim3(1, 1, 1), dim3(32, 1, 1), streams[2] >> >(utility + 2, new_boards[2], t_alpha, beta, depth - 1, turns - 1);
				white_min_kernel << <dim3(1, 1, 1), dim3(32, 1, 1), streams[3] >> >(utility + 3, new_boards[3], t_alpha, beta, depth - 1, turns - 1);


				// sync streams here

				for (int i = 0; i < 4; ++i)
				{
					cudaStreamSynchronize(streams[i]);
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
				cudaFree(new_boards);

				cudaStreamDestroy(streams[0]);
				cudaStreamDestroy(streams[1]);
				cudaStreamDestroy(streams[2]);
				cudaStreamDestroy(streams[3]);

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

		__global__ void master_black_max_kernel(utility_type *v, utility_type *utility, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns)
		{
			int tx = threadIdx.x;
			int index = blockIdx.x * blockDim.x + tx;
			int t_v = -Minimax::Infinity;

			if (index < num_boards)
			{
				black_min_kernel << <dim3(1, 1, 1), dim3(32, 1, 1) >> > (utility + index, src[index], alpha, beta, depth - 1, turns - 1);

			}
			__syncthreads();

			if (!index)
			{
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

		__global__ void master_black_min_kernel(utility_type *v, utility_type *utility, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns)
		{
			int tx = threadIdx.x;
			int index = blockIdx.x * blockDim.x + tx;
			int t_v = Minimax::Infinity;

			if (index < num_boards)
			{
				black_max_kernel << <dim3(1, 1, 1), dim3(32, 1, 1) >> > (utility + index, src + index, alpha, beta, depth - 1, turns - 1);

			}
			__syncthreads();

			if (index == 0)
			{
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

		__global__ void black_min_kernel(utility_type *v, GPUBitBoard src, int alpha, int beta, int depth, int turns)
		{
			int tx = threadIdx.x;
			int t_beta = beta;
			int t_v = Minimax::Infinity;
			__shared__ bool terminated;
			__shared__ utility_type utilities[32];

			if (tx == 0)
			{
				utility_type terminal_value = 0;
				if (src->valid)
				{
					terminated = GPUGetWhiteUtility(src, &terminal_value, depth, turns);
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

				utility_type * utility;
				cudaMalloc(&utility, sizeof(utility_type) * 4);
				utility[0] = utility[1] = utility[2] = utility[3] = utilities[tx];
				GPUBitBoard *new_boards;
				cudaMalloc(&new_boards, sizeof(GPUBitBoard) * 4);

				cudaStream_t streams[4];
				cudaStreamCreateWithFlags(&streams[0], cudaStreamNonBlocking);
				cudaStreamCreateWithFlags(&streams[1], cudaStreamNonBlocking);
				cudaStreamCreateWithFlags(&streams[2], cudaStreamNonBlocking);
				cudaStreamCreateWithFlags(&streams[3], cudaStreamNonBlocking);

				// in the max kernel, use gen_black_move_type instead
				int gen_White_move_type = (int)GPUBitBoard::GetWhiteJumps(src) != 0);
				gen_white_move[gen_White_move_type](1u << tx, new_board, src);

				black_max_kernel << <dim3(1, 1, 1), dim3(32, 1, 1), streams[0] >> >(utility, new_boards[0], alpha, t_beta, depth - 1, turns - 1);
				black_max_kernel << <dim3(1, 1, 1), dim3(32, 1, 1), streams[1] >> >(utility + 1, new_boards[1], alpha, t_beta, depth - 1, turns - 1);
				black_max_kernel << <dim3(1, 1, 1), dim3(32, 1, 1), streams[2] >> >(utility + 2, new_boards[2], alpha, t_beta, depth - 1, turns - 1);
				black_max_kernel << <dim3(1, 1, 1), dim3(32, 1, 1), streams[3] >> >(utility + 3, new_boards[3], alpha, t_beta, depth - 1, turns - 1);


				// sync streams here

				for (int i = 0; i < 4; ++i)
				{
					cudaStreamSynchronize(streams[i]);
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
				cudaFree(new_boards);

				cudaStreamDestroy(streams[0]);
				cudaStreamDestroy(streams[1]);
				cudaStreamDestroy(streams[2]);
				cudaStreamDestroy(streams[3]);

				__syncthreads();

				if (tx == 0)
				{
					// final ab-pruning for this node
					for (int i = 0; i < 32; ++i)
					{
						t_v = min(utilities[i], t_v);
						if (t_v < alpha))
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

		__global__ void black_max_kernel(utility_type *v, GPUBitBoard src, int alpha, int beta, int depth, int turns)
		{
			int tx = threadIdx.x;
			int t_alpha = alpha;
			int t_v = -Minimax::Infinity;
			__shared__ bool terminated;
			__shared__ utility_type utilities[32];

			if (tx == 0)
			{
				utility_type terminal_value = 0;
				if (src->valid)
				{
					terminated = GPUGetWhiteUtility(src, &terminal_value, depth, turns);
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

				utility_type * utility;
				cudaMalloc(&utility, sizeof(utility_type) * 4);
				utility[0] = utility[1] = utility[2] = utility[3] = utilities[tx];
				GPUBitBoard *new_boards;
				cudaMalloc(&new_boards, sizeof(GPUBitBoard) * 4);

				cudaStream_t streams[4];
				cudaStreamCreateWithFlags(&streams[0], cudaStreamNonBlocking);
				cudaStreamCreateWithFlags(&streams[1], cudaStreamNonBlocking);
				cudaStreamCreateWithFlags(&streams[2], cudaStreamNonBlocking);
				cudaStreamCreateWithFlags(&streams[3], cudaStreamNonBlocking);

				// in the max kernel, use gen_black_move_type instead
				int gen_black_move_type = (int)GPUBitBoard::GetBlackJumps(src) != 0);
				gen_black_move[gen_black_move_type](1u << tx, new_board, src);

				black_min_kernel << <dim3(1, 1, 1), dim3(32, 1, 1), streams[0] >> >(utility, new_boards[0], t_alpha, beta, depth - 1, turns - 1);
				black_min_kernel << <dim3(1, 1, 1), dim3(32, 1, 1), streams[1] >> >(utility + 1, new_boards[1], t_alpha, beta, depth - 1, turns - 1);
				black_min_kernel << <dim3(1, 1, 1), dim3(32, 1, 1), streams[2] >> >(utility + 2, new_boards[2], t_alpha, beta, depth - 1, turns - 1);
				black_min_kernel << <dim3(1, 1, 1), dim3(32, 1, 1), streams[3] >> >(utility + 3, new_boards[3], t_alpha, beta, depth - 1, turns - 1);


				// sync streams here

				for (int i = 0; i < 4; ++i)
				{
					cudaStreamSynchronize(streams[i]);
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
				cudaFree(new_boards);

				cudaStreamDestroy(streams[0]);
				cudaStreamDestroy(streams[1]);
				cudaStreamDestroy(streams[2]);
				cudaStreamDestroy(streams[3]);

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

		__device__ bool GPUGetWhiteUtility(GPUBitBoard const &src, utility_type &terminal_value, int depth, int turns)
		{

		}

		__device__ bool GPUGetBlackUtility(GPUBitBoard const &src, utility_type &terminal_value, int depth, int turns)
		{

		}

		// define the host functions here as well because it needs to run the kernels

		bool WhiteWinTest(GPUBitBoard const &b)
		{

		}

		bool WhiteLoseTest(GPUBitBoard const &b)
		{

		}

		bool BlackWinTest(GPUBitBoard const &b)
		{

		}

		bool BlackLoseTest(GPUBitBoard const &b)
		{

		}


		bool GetBlackUtility(GPUBitBoard const &b, utility_type &utility, int depth, int turns_left)
		{

		}

		bool GetWhiteUtility(GPUBitBoard const &b, utility_type &utility, int depth, int turns_left)
		{

		}

		utility_type WhiteMoveMax(GPUBitBoard const &b, int depth, int turns_left, utility_type alpha, utility_type beta)
		{

		}

		utility_type WhiteMoveMin(GPUBitBoard const &b, int depth, int turns_left, utility_type alpha, utility_type beta)
		{

		}

		utility_type BlackMoveMax(GPUBitBoard const &b, int depth, int turns_left, utility_type alpha, utility_type beta)
		{

		}

		utility_type BlackMoveMin(GPUBitBoard const &b, int depth, int turns_left, utility_type alpha, utility_type beta)
		{

		}
	}
}