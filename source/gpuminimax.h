#pragma once

#include "minimax.h"

namespace Checkers
{
	namespace GPUMinimax
	{
		using utility_type = Minimax::utility_type;

		static bool WhiteWinTest(BitBoard const &b);
		static bool WhiteLoseTest(BitBoard const &b);
		static bool BlackWinTest(BitBoard const &b);
		static bool BlackLoseTest(BitBoard const &b);

		// utility functions should be called for terminal test (i.e. before frontier generation).
		static bool GetBlackUtility(BitBoard const &b, utility_type &utility, int depth, int turns_left);
		static bool GetWhiteUtility(BitBoard const &b, utility_type &utility, int depth, int turns_left);

		static utility_type WhiteMoveMax(BitBoard const &b, int depth, int turns_left,utility_type alpha, utility_type beta);
		static utility_type WhiteMoveMin(BitBoard const &b, int depth, int turns_left,utility_type alpha, utility_type beta);
		static utility_type BlackMoveMax(BitBoard const &b, int depth, int turns_left,utility_type alpha, utility_type beta);
		static utility_type BlackMoveMin(BitBoard const &b, int depth, int turns_left,utility_type alpha, utility_type beta);
	}
}