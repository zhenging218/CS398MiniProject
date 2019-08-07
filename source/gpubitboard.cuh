#pragma once

#include "minimax.h"
#include <thrust/system/cuda/config.h>

namespace Checkers
{
	struct GPUBitBoard
	{
		using utility_type = Minimax::utility_type;
		using count_type = BitBoard::count_type;
		using board_type = BitBoard::board_type;
		using count_type = BitBoard::count_type;

		using gen_move_func = void(*)(GPUBitBoard::board_type, GPUBitBoard *, GPUBitBoard const *);

		board_type white, black, kings;
		bool valid;

		__device__ static GPUBitBoard::gen_move_func gen_white_move[2];
		__device__ static GPUBitBoard::gen_move_func gen_black_move[2];

		__host__ __device__ GPUBitBoard();
		__host__ __device__ GPUBitBoard(board_type w, board_type b, board_type k);
		__host__ __device__ GPUBitBoard &operator=(GPUBitBoard const &src);

		__host__ GPUBitBoard(BitBoard const &src);
		__host__ GPUBitBoard &operator=(BitBoard const &src);

		__host__ operator BitBoard() const;

		__host__ __device__ static board_type GetBlackMoves(BitBoard const &b);
		__host__ __device__ static board_type GetWhiteMoves(BitBoard const &b);

		__host__ __device__ static board_type GetBlackJumps(GPUBitBoard const &b);
		__host__ __device__ static board_type GetWhiteJumps(GPUBitBoard const &b);

		__device__ static count_type GetBlackPieceCount(GPUBitBoard const &b);
		__device__ static count_type GetWhitePieceCount(GPUBitBoard const &b);

		__device__ static count_type GetBlackKingsCount(GPUBitBoard const &b);
		__device__ static count_type GetWhiteKingsCount(GPUBitBoard const &b);

		__device__ void GenWhiteMove(board_type cell, GPUBitBoard *out, GPUBitBoard const &board);
		__device__ void GenWhiteJump(board_type cell, GPUBitBoard *out, GPUBitBoard const &board);
		__device__ void GenBlackMove(board_type cell, GPUBitBoard *out, GPUBitBoard const &board);
		__device__ void GenBlackJump(board_type cell, GPUBitBoard *out, GPUBitBoard const &board)
	};
}