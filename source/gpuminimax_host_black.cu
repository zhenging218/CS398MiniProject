#include "precomp.cuh"
#include "bitboard.h"
#include "gpuminimax.h"

#include <cstdlib>
#include <algorithm>

namespace Checkers
{
	namespace GPUMinimax
	{
		__host__ __device__ Minimax::utility_type BlackMoveMax(BitBoard const &b, int depth, int turns_left, Minimax::utility_type alpha, Minimax::utility_type beta)
		{
			utility_type v = -Infinity;
			utility_type terminal_value = 0;
			// check if need to stop the search
			if (GetBlackUtility(b, terminal_value, depth, turns_left))
				return terminal_value;

			BitBoard frontier[32];
			BitBoard *end = frontier;
			BitBoard::GetPossibleBlackMoves(b, end);
			int size = end - frontier;

			if (size > 0)
			{
				v = max(BlackMoveMin(frontier[0], depth - 1, turns_left - 1, alpha, beta), v);
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
						master_black_max_kernel(GPUv, GPUFrontier, size - 1, alpha, beta, depth - 1, turns_left - 1);
						CHECK_ERRORS();

						cudaMemcpy(&v, GPUv, sizeof(int), cudaMemcpyDeviceToHost);
						cudaFree(GPUFrontier);
						CHECK_ERRORS();
						cudaFree(GPUv);
						CHECK_ERRORS();

					}
				}
			}

			return v;
		}

		__host__ __device__ Minimax::utility_type BlackMoveMin(BitBoard const &b, int depth, int turns_left, Minimax::utility_type alpha, Minimax::utility_type beta)
		{
			utility_type v = Infinity;
			utility_type terminal_value = 0;
			// check if need to stop the search
			if (GetBlackUtility(b, terminal_value, depth, turns_left))
				return terminal_value;

			BitBoard frontier[32];
			BitBoard *end = frontier;
			BitBoard::GetPossibleWhiteMoves(b, end);
			int size = end - frontier;

			if (size > 0)
			{
				v = min(BlackMoveMax(frontier[0], depth - 1, turns_left - 1, alpha, beta), v);
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
						CHECK_ERRORS();
						cudaMemcpy(GPUFrontier, copy, sizeof(GPUBitBoard) * (size - 1), cudaMemcpyHostToDevice);
						CHECK_ERRORS();
						free(copy);

						cudaMalloc((void**)&GPUv, sizeof(utility_type));
						CHECK_ERRORS();
						cudaMemcpy(GPUv, &v, sizeof(utility_type), cudaMemcpyHostToDevice);
						CHECK_ERRORS();

						// launch kernel
						master_black_min_kernel(GPUv, GPUFrontier, size - 1, alpha, beta, depth - 1, turns_left - 1);
						CHECK_ERRORS();

						cudaMemcpy(&v, GPUv, sizeof(int), cudaMemcpyDeviceToHost);
						CHECK_ERRORS();
						cudaFree(GPUFrontier);
						CHECK_ERRORS();
						cudaFree(GPUv);
						CHECK_ERRORS();

					}
				}
			}

			return v;
		}
	}
}