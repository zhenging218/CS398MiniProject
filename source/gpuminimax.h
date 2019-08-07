#pragma once

#include "bitboard.h"
#include "minimax.h"

namespace Checkers
{
	namespace GPUMinimax
	{
		using utility_type = Minimax::utility_type;

		BitBoard Next(BitBoard const &board, Minimax::Turn &turn, int depth, int &turns_left);
	}
}