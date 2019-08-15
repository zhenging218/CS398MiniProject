#include "precomp.cuh"
#include "bitboard.h"
#include "gpuminimax.h"

#include <helper_cuda.h>

#include <cstdlib>
#include <algorithm>

namespace Checkers
{
	namespace GPUMinimax
	{
		__device__ GPUBitBoard::gen_move_func gen_white_move[2] = { GPUBitBoard::GenWhiteMove, GPUBitBoard::GenWhiteJump };
		__device__ GPUBitBoard::gen_move_func gen_black_move[2] = { GPUBitBoard::GenBlackMove, GPUBitBoard::GenBlackJump };

		__device__ extern GPUBitBoard::gen_move_atomic_func gen_white_move_atomic[2] = { GPUBitBoard::GenWhiteMoveAtomic, GPUBitBoard::GenWhiteJumpAtomic };
		__device__ extern GPUBitBoard::gen_move_atomic_func gen_black_move_atomic[2] = { GPUBitBoard::GenBlackMoveAtomic, GPUBitBoard::GenBlackJumpAtomic };

		__host__ __device__ NodeType &operator++(NodeType &src)
		{
			src = (src == NodeType::MAX) ? NodeType::MIN : NodeType::MAX;
			return src;
		}

		__host__ __device__ NodeType operator+(NodeType const &src, int i)
		{
			NodeType ret = src;
			while (i > 0)
			{
				++ret;
				--i;
			}
			return ret;
		}


		Checkers::Minimax::Result Next(BitBoard &board, Checkers::Minimax::Turn &turn, int depth, int &turns_left)
		{
			if (turns_left == 0)
			{
				return Minimax::DRAW;
			}

			int placement = -1;
			utility_type X = -Infinity;
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
				//Minimax::utility_type X = -Minimax::Infinity;
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
					utility_type *GPUUtility;
					int * GPUPlacement;

					GPUBitBoard *copy = (GPUBitBoard*)malloc(sizeof(GPUBitBoard) * (size - 1));
					for (int i = 0; i < (size - 1); ++i)
					{
						new (copy + i) GPUBitBoard(frontier[i + 1]);
					}
					cudaMalloc((void**)&GPUFrontier, sizeof(GPUBitBoard) * (size - 1));
					CHECK_ERRORS();

					cudaMemcpy(GPUFrontier, copy, sizeof(GPUBitBoard) * (size - 1), cudaMemcpyHostToDevice);
					CHECK_ERRORS();

					free(copy);

					cudaMalloc((void**)&GPUUtility, sizeof(utility_type) * (size - 1));
					CHECK_ERRORS();

					cudaMalloc((void**)&GPUPlacement, sizeof(int));
					CHECK_ERRORS();

					cudaMemcpy(GPUPlacement, &placement, sizeof(int), cudaMemcpyHostToDevice);
					CHECK_ERRORS();

					// launch kernel
					white_next_kernel << < dim3(size - 1, 1, 1), dim3(32, 1, 1) >> > (GPUPlacement, GPUUtility, X, GPUFrontier, size - 1, depth, turns_left);
					cudaDeviceSynchronize();
					CHECK_ERRORS();

					cudaMemcpy(&placement, GPUPlacement, sizeof(int), cudaMemcpyDeviceToHost);
					CHECK_ERRORS();
					cudaFree(GPUFrontier);
					CHECK_ERRORS();
					cudaFree(GPUUtility);
					CHECK_ERRORS();
					cudaFree(GPUPlacement);
					CHECK_ERRORS();

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
					utility_type *GPUUtility;
					int * GPUPlacement;

					GPUBitBoard *copy = (GPUBitBoard*)malloc(sizeof(GPUBitBoard) * (size - 1));
					for (int i = 0; i < (size - 1); ++i)
					{
						new (copy + i) GPUBitBoard(frontier[i + 1]);
					}
					cudaMalloc((void**)&GPUFrontier, sizeof(GPUBitBoard) * (size - 1));
					CHECK_ERRORS();

					cudaMemcpy(GPUFrontier, copy, sizeof(GPUBitBoard) * (size - 1), cudaMemcpyHostToDevice);
					CHECK_ERRORS();

					free(copy);

					cudaMalloc((void**)&GPUUtility, sizeof(utility_type) * (size - 1));
					CHECK_ERRORS();

					cudaMalloc((void**)&GPUPlacement, sizeof(int));
					CHECK_ERRORS();

					cudaMemcpy(GPUPlacement, &placement, sizeof(int), cudaMemcpyHostToDevice);
					CHECK_ERRORS();

					// launch kernel
					black_next_kernel << < dim3(size - 1, 1, 1), dim3(32, 1, 1) >> > (GPUPlacement, GPUUtility, X, GPUFrontier, size - 1, depth, turns_left);
					cudaDeviceSynchronize();
					CHECK_ERRORS();

					cudaMemcpy(&placement, GPUPlacement, sizeof(int), cudaMemcpyDeviceToHost);
					CHECK_ERRORS();

					cudaFree(GPUFrontier);
					CHECK_ERRORS();
					cudaFree(GPUUtility);
					CHECK_ERRORS();
					cudaFree(GPUPlacement);
					CHECK_ERRORS();
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