#include "precomp.cuh"
#include "bitboard.h"
#include "gpuminimax.h"

namespace Checkers
{
	// all master kernels should already have v and the utility array values initialised to the first v value computed by the CPU PV-Split.

	namespace GPUMinimax
	{
		__device__ utility_type white_min_device(GPUBitBoard src, int alpha, int beta, int depth, int turns)
		{
			utility_type v = Infinity;
			utility_type terminal_value = 0;
			// check if need to stop the search
			if (GetWhiteUtility(b, terminal_value, depth, turns_left))
				return terminal_value;

			auto frontier = GenerateBlackFrontier(b);

			for (auto const &move : frontier)
			{
				v = min(WhiteMoveMax(move, depth - 1, turns_left - 1, alpha, beta), v);
				if (v < alpha)
				{
					// prune
					break;
				}
				beta = min(beta, v);
			}

			return v;
		}

		__device__ utility_type white_max_device(GPUBitBoard src, int alpha, int beta, int depth, int turns)
		{
			utility_type v = -Infinity;
			utility_type terminal_value = 0;
			// check if need to stop the search
			if (GetWhiteUtility(b, terminal_value, depth, turns_left))
				return terminal_value;

			auto frontier = GenerateWhiteFrontier(b);

			for (auto const &move : frontier)
			{
				v = max(WhiteMoveMin(move, depth - 1, turns_left - 1, alpha, beta), v);
				if (v > beta)
				{
					// prune
					break;
				}
				alpha = std::max(alpha, v);
			}

			return v;
		}
}