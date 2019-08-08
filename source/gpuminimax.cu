#include "precomp.cuh"
#include "bitboard.h"
#include "gpuminimax.h"

#include <cstdlib>
#include <algorithm>

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
	}
}