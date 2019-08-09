#include "precomp.cuh"
#include "bitboard.h"
#include "gpuminimax.h"

namespace Checkers
{
	namespace GPUMinimax
	{
		__device__ utility_type black_min_device(GPUBitBoard src, int alpha, int beta, int depth, int turns)
		{
			utility_type v = Infinity;
			utility_type terminal_value = 0;
			int i = 0;
			GPUBitBoard moves[128];
			// check if need to stop the search
			//if (GetBlackUtility(src, terminal_value, depth, turns))
			//	return terminal_value;
			//// 32 possible moves
			//while()
			//	GenBlackMove(i,moves,src)
			//}

			//for (auto const &move : frontier)
			//{
			//	v = min(black_max_device(move, depth - 1, turns - 1, alpha, beta), v);
			//	if (v < alpha)
			//	{
			//		// prune
			//		break;
			//	}
			//	beta = min(beta, v);
			//}

			return v;
		}

		__device__ utility_type black_max_device(GPUBitBoard src, int alpha, int beta, int depth, int turns)
		{
			utility_type v = -Infinity;
			utility_type terminal_value = 0;
			GPUBitBoard moves[128];
			// check if need to stop the search
			if (GetBlackUtility(src, terminal_value, depth, turns))
				return terminal_value;

			// 32 possible moves
			for (int i = 0; i < 32; i++)
			{
				GenBlackMove(i, moves, src)
			}

			for (auto const &move : frontier)
			{
				v = max(BlackMoveMin(move, depth - 1, turns - 1, alpha, beta), v);
				if (v > beta)
				{
					// prune
					break;
				}
				alpha = max(alpha, v);
			}

			return v;
		}
	}
}