#pragma once

#include "bitboard.h"
#include "minimax.h"

namespace Checkers
{
	struct GPUBitBoard
	{
		using board_type = BitBoard::board_type;
		using count_type = BitBoard::count_type;

		board_type white, black, kings;

		__host__ __device__ GPUBitBoard();
		__host__ __device__ GPUBitBoard(board_type w, board_type b, board_type k);
		__host__ __device__ GPUBitBoard &operator=(GPUBitBoard const &src);

		__host__ GPUBitBoard(BitBoard const &src);
		__host__ GPUBitBoard &operator=(BitBoard const &src);

		__host__ operator BitBoard() const;

		__device__ static board_type GetBlackMoves(GPUBitBoard const &b);
		__device__ static board_type GetWhiteMoves(GPUBitBoard const &b);

		__device__ static board_type GetBlackJumps(GPUBitBoard const &b);
		__device__ static board_type GetWhiteJumps(GPUBitBoard const &b);

		__device__ static count_type GetBlackPieceCount(GPUBitBoard const &b);
		__device__ static count_type GetWhitePieceCount(GPUBitBoard const &b);

		__device__ static count_type GetBlackKingsCount(GPUBitBoard const &b);
		__device__ static count_type GetWhiteKingsCount(GPUBitBoard const &b);
	};
}