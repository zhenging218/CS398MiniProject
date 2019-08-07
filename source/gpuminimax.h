#pragma once

#include "minimax.h"

namespace Checkers
{
	namespace GPUMinimax
	{
		Minimax::Result Next(BitBoard &board, Minimax::Turn &turn, int depth, int &turns_left);
	}
}