#pragma once

#include "bitboard.h"
#include "minimax.h"

namespace Checkers
{
	BitBoard RunGPUMinimax(BitBoard const &src, Minimax::Turn turn, int &turns_left, int max_turns);
}