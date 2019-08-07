#include "precomp.cuh"
#include "bitboard.h"
#include "gpuminimax.h"

#include <cstdlib>
#include <algorithm>
#include <helper_cuda.h>
#include <cuda_runtime.h>

namespace Checkers
{
	namespace GPUMinimax
	{
		__device__ GPUBitBoard::gen_move_func gen_white_move[2] = { GPUBitBoard::GenWhiteMove, GPUBitBoard::GenWhiteJump };
		__device__ GPUBitBoard::gen_move_func gen_black_move[2] = { GPUBitBoard::GenBlackMove, GPUBitBoard::GenBlackJump };

		Checkers::Minimax::Result Next(BitBoard &board, Checkers::Minimax::Turn &turn, int depth, int &turns_left)
		{
			if (turns_left == 0)
			{
				return Minimax::DRAW;
			}

			BitBoard frontier[32];

			if (turn == Minimax::WHITE)
			{
				BitBoard *end = frontier;
				BitBoard::GetPossibleWhiteMoves(board, end);
				int size = end - frontier;
				if (size == 0)
				{
					return Minimax::LOSE;
				}

				int placement = -1;
				Minimax::utility_type X = -Minimax::Infinity;

				// CPU left-most branch
				Minimax::utility_type v = WhiteMoveMin(frontier[0], depth, turns_left, -Infinity, Infinity);
				if (X < v)
				{
					X = v;
					placement = 0;
				}

				if (size > 1)
				{
					// GPU tree-split the rest of the branches
					GPUBitBoard * GPUFrontier;
					int * GPUPlacement;

					GPUBitBoard *copy = (GPUBitBoard*)malloc(sizeof(GPUBitBoard) * (size - 1));
					for (int i = 0; i < (size - 1); ++i)
					{
						new (copy + i) GPUBitBoard(frontier[i + 1]);
					}
					cudaMalloc((void**)&GPUFrontier, sizeof(GPUBitBoard) * (size - 1));
					cudaMemcpy(GPUFrontier, copy, sizeof(GPUBitBoard) * (size - 1), cudaMemcpyHostToDevice);
					free(copy);

					cudaMalloc((void**)&GPUPlacement, sizeof(int));
					cudaMemcpy(GPUPlacement, &placement, sizeof(int), cudaMemcpyHostToDevice);

					// launch kernel
					master_white_next_kernel << < dim3(((size - 1) / 32) + 1, 1, 1), dim3(32, 1, 1) >> > (GPUPlacement, X, GPUFrontier, size - 1, depth, turns_left);
					cudaDeviceSynchronize();

					cudaMemcpy(&placement, GPUPlacement, sizeof(int), cudaMemcpyDeviceToHost);
					cudaFree(GPUFrontier);
					cudaFree(GPUPlacement);
				}

				if (placement >= 0)
				{
					board = frontier[placement];
				}
			}
			else
			{
				BitBoard *end = frontier;
				BitBoard::GetPossibleBlackMoves(board, end);
				int size = end - frontier;
				if (size == 0)
				{
					return Minimax::Result::LOSE;
				}

				int placement = -1;
				Minimax::utility_type X = -Minimax::Infinity;

				// CPU left-most branch
				Minimax::utility_type v = BlackMoveMin(frontier[0], depth, turns_left, -Infinity, Infinity);
				if (X < v)
				{
					X = v;
					placement = 0;
				}

				if (size > 1)
				{
					// GPU tree-split the rest of the branches
					GPUBitBoard * GPUFrontier;
					int * GPUPlacement;

					GPUBitBoard *copy = (GPUBitBoard*)malloc(sizeof(GPUBitBoard) * (size - 1));
					for (int i = 0; i < (size - 1); ++i)
					{
						new (copy + i) GPUBitBoard(frontier[i + 1]);
					}
					cudaMalloc((void**)&GPUFrontier, sizeof(GPUBitBoard) * (size - 1));
					cudaMemcpy(GPUFrontier, copy, sizeof(GPUBitBoard) * (size - 1), cudaMemcpyHostToDevice);
					free(copy);

					cudaMalloc((void**)&GPUPlacement, sizeof(int));
					cudaMemcpy(GPUPlacement, &placement, sizeof(int), cudaMemcpyHostToDevice);

					// launch kernel
					master_black_next_kernel << <dim3(1, 1, 1), dim3(32, 1, 1) >> > (GPUPlacement, X, GPUFrontier, size - 1, depth, turns_left);
					cudaDeviceSynchronize();

					cudaMemcpy(&placement, GPUPlacement, sizeof(int), cudaMemcpyDeviceToHost);
					cudaFree(GPUFrontier);
					cudaFree(GPUPlacement);
				}

				if (placement >= 0)
				{
					board = frontier[placement];
				}
			}

			++turn;
			if (turns_left)
			{
				--turns_left;
			}

			return Minimax::INPROGRESS;
		}

		__host__ Minimax::utility_type WhiteMoveMax(BitBoard const &b, int depth, int turns_left, Minimax::utility_type alpha, Minimax::utility_type beta)
		{
			utility_type v = -Infinity;
			utility_type terminal_value = 0;
			// check if need to stop the search
			if (GetWhiteUtility(b, terminal_value, depth, turns_left))
				return terminal_value;

			BitBoard frontier[32];
			BitBoard *end = frontier;
			BitBoard::GetPossibleWhiteMoves(b, end);
			int size = end - frontier;
			
			if (size > 0)
			{
				v = std::max(WhiteMoveMin(frontier[0], depth - 1, turns_left - 1, alpha, beta), v);
				if (!(v > beta))
				{
					alpha = std::max(alpha, v);
					if (size > 1)
					{
						GPUBitBoard * GPUFrontier;
						utility_type * GPUv;

						GPUBitBoard *copy = (GPUBitBoard*)malloc(sizeof(GPUBitBoard) * (size - 1));
						for (int i = 0; i < (size - 1); ++i)
						{
							new (copy + i) GPUBitBoard(frontier[i + 1]);
						}
						cudaMalloc((void**)&GPUFrontier, sizeof(GPUBitBoard) * (size - 1));
						cudaMemcpy(GPUFrontier, copy, sizeof(GPUBitBoard) * (size - 1), cudaMemcpyHostToDevice);
						free(copy);

						cudaMalloc((void**)&GPUv, sizeof(utility_type));
						cudaMemcpy(GPUv, &v, sizeof(utility_type), cudaMemcpyHostToDevice);

						// launch kernel
						master_white_min_kernel << <dim3(1, 1, 1), dim3(32, 1, 1) >> > (GPUv, GPUFrontier, size - 1, alpha, beta, depth - 1, turns_left - 1);
						cudaDeviceSynchronize();

						cudaMemcpy(&v, GPUv, sizeof(int), cudaMemcpyDeviceToHost);
						cudaFree(GPUFrontier);
						cudaFree(GPUv);
					}
				}
			}

			return v;
		}

		__host__ Minimax::utility_type WhiteMoveMin(BitBoard const &b, int depth, int turns_left, Minimax::utility_type alpha, Minimax::utility_type beta)
		{
			utility_type v = Infinity;
			utility_type terminal_value = 0;
			// check if need to stop the search
			if (GetWhiteUtility(b, terminal_value, depth, turns_left))
				return terminal_value;

			BitBoard frontier[32];
			BitBoard *end = frontier;
			BitBoard::GetPossibleBlackMoves(b, end);
			int size = end - frontier;

			if (size > 0)
			{
				v = std::max(WhiteMoveMax(frontier[0], depth - 1, turns_left - 1, alpha, beta), v);
				if (!(v < alpha))
				{
					beta = std::min(beta, v);
					if (size > 1)
					{
						GPUBitBoard * GPUFrontier;
						utility_type * GPUv;

						GPUBitBoard *copy = (GPUBitBoard*)malloc(sizeof(GPUBitBoard) * (size - 1));
						for (int i = 0; i < (size - 1); ++i)
						{
							new (copy + i) GPUBitBoard(frontier[i + 1]);
						}
						cudaMalloc((void**)&GPUFrontier, sizeof(GPUBitBoard) * (size - 1));
						cudaMemcpy(GPUFrontier, copy, sizeof(GPUBitBoard) * (size - 1), cudaMemcpyHostToDevice);
						free(copy);

						cudaMalloc((void**)&GPUv, sizeof(utility_type));
						cudaMemcpy(GPUv, &v, sizeof(utility_type), cudaMemcpyHostToDevice);

						// launch kernel
						master_white_max_kernel << <dim3(1, 1, 1), dim3(32, 1, 1) >> > (GPUv, GPUFrontier, size - 1, alpha, beta, depth - 1, turns_left - 1);
						cudaDeviceSynchronize();

						cudaMemcpy(&v, GPUv, sizeof(int), cudaMemcpyDeviceToHost);
						cudaFree(GPUFrontier);
						cudaFree(GPUv);
					}
				}
			}

			return v;
		}

		__host__ Minimax::utility_type BlackMoveMax(BitBoard const &b, int depth, int turns_left, Minimax::utility_type alpha, Minimax::utility_type beta)
		{
			utility_type v = -Infinity;
			utility_type terminal_value = 0;
			// check if need to stop the search
			if (GetWhiteUtility(b, terminal_value, depth, turns_left))
				return terminal_value;

			BitBoard frontier[32];
			BitBoard *end = frontier;
			BitBoard::GetPossibleBlackMoves(b, end);
			int size = end - frontier;

			if (size > 0)
			{
				v = std::max(BlackMoveMin(frontier[0], depth - 1, turns_left - 1, alpha, beta), v);
				if (!(v > beta))
				{
					alpha = std::max(alpha, v);
					if (size > 1)
					{
						GPUBitBoard * GPUFrontier;
						utility_type * GPUv;

						GPUBitBoard *copy = (GPUBitBoard*)malloc(sizeof(GPUBitBoard) * (size - 1));
						for (int i = 0; i < (size - 1); ++i)
						{
							new (copy + i) GPUBitBoard(frontier[i + 1]);
						}
						cudaMalloc((void**)&GPUFrontier, sizeof(GPUBitBoard) * (size - 1));
						cudaMemcpy(GPUFrontier, copy, sizeof(GPUBitBoard) * (size - 1), cudaMemcpyHostToDevice);
						free(copy);

						cudaMalloc((void**)&GPUv, sizeof(utility_type));
						cudaMemcpy(GPUv, &v, sizeof(utility_type), cudaMemcpyHostToDevice);

						// launch kernel
						master_black_min_kernel << <dim3(1, 1, 1), dim3(32, 1, 1) >> > (GPUv, GPUFrontier, size - 1, alpha, beta, depth - 1, turns_left - 1);
						cudaDeviceSynchronize();

						cudaMemcpy(&v, GPUv, sizeof(int), cudaMemcpyDeviceToHost);
						cudaFree(GPUFrontier);
						cudaFree(GPUv);
					}
				}
			}

			return v;
		}

		__host__ Minimax::utility_type BlackMoveMin(BitBoard const &b, int depth, int turns_left, Minimax::utility_type alpha, Minimax::utility_type beta)
		{
			utility_type v = Infinity;
			utility_type terminal_value = 0;
			// check if need to stop the search
			if (GetWhiteUtility(b, terminal_value, depth, turns_left))
				return terminal_value;

			BitBoard frontier[32];
			BitBoard *end = frontier;
			BitBoard::GetPossibleWhiteMoves(b, end);
			int size = end - frontier;

			if (size > 0)
			{
				v = std::max(BlackMoveMax(frontier[0], depth - 1, turns_left - 1, alpha, beta), v);
				if (!(v < alpha))
				{
					beta = std::min(beta, v);
					if (size > 1)
					{
						GPUBitBoard * GPUFrontier;
						utility_type * GPUv;

						GPUBitBoard *copy = (GPUBitBoard*)malloc(sizeof(GPUBitBoard) * (size - 1));
						for (int i = 0; i < (size - 1); ++i)
						{
							new (copy + i) GPUBitBoard(frontier[i + 1]);
						}
						cudaMalloc((void**)&GPUFrontier, sizeof(GPUBitBoard) * (size - 1));
						cudaMemcpy(GPUFrontier, copy, sizeof(GPUBitBoard) * (size - 1), cudaMemcpyHostToDevice);
						free(copy);

						cudaMalloc((void**)&GPUv, sizeof(utility_type));
						cudaMemcpy(GPUv, &v, sizeof(utility_type), cudaMemcpyHostToDevice);

						// launch kernel
						master_black_max_kernel << <dim3(1, 1, 1), dim3(32, 1, 1) >> > (GPUv, GPUFrontier, size - 1, alpha, beta, depth - 1, turns_left - 1);
						cudaDeviceSynchronize();

						cudaMemcpy(&v, GPUv, sizeof(int), cudaMemcpyDeviceToHost);
						cudaFree(GPUFrontier);
						cudaFree(GPUv);
					}
				}
			}

			return v;
		}
	}
}