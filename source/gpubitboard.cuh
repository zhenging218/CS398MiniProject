#pragma once

#include "bitboard.h"
#include "minimax.h"
#include <helper_cuda.h>

namespace Checkers
{
	struct GPUBitBoard
	{
		using board_type = BitBoard::board_type;
		using utility_type = Minimax::utility_type;

		board_type white, black, kings;
		bool valid;

		__host__ __device__ GPUBitBoard();
		__host__ __device__ GPUBitBoard(board_type w, board_type b, board_type k, bool v);
		__host__ __device__ GPUBitBoard &operator=(GPUBitBoard const &src);

		__host__ GPUBitBoard(BitBoard const &src);
		__host__ GPUBitBoard &operator=(BitBoard const &src);

		__host__ operator BitBoard() const;
	};
}