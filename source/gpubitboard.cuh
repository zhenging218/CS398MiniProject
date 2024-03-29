#pragma once

#include "minimax.h"

namespace Checkers
{
	struct GPUBitBoard
	{
		using utility_type = Minimax::Minimax::utility_type;
		using count_type = BitBoard::count_type;
		using board_type = BitBoard::board_type;

		static constexpr GPUBitBoard::board_type Row0 = 0x10000000u;

		static constexpr GPUBitBoard::board_type L3Mask = BitBoard::L3Mask;
		static constexpr GPUBitBoard::board_type L5Mask = BitBoard::L5Mask;
		static constexpr GPUBitBoard::board_type R3Mask = BitBoard::R3Mask;
		static constexpr GPUBitBoard::board_type R5Mask = BitBoard::R5Mask;

		static constexpr GPUBitBoard::board_type OddRows = BitBoard::OddRows;

		static constexpr GPUBitBoard::board_type BlackKingMask = BitBoard::BlackKingMask;
		static constexpr GPUBitBoard::board_type WhiteKingMask = BitBoard::WhiteKingMask;

		using gen_move_func = void(*)(GPUBitBoard::board_type, GPUBitBoard const &, GPUBitBoard *, int &);
		using gen_move_atomic_func = void(*)(GPUBitBoard::board_type, GPUBitBoard const &, GPUBitBoard *, int *);

		board_type white, black, kings;

		__host__ __device__ GPUBitBoard();
		__host__ __device__ GPUBitBoard(board_type w, board_type b, board_type k);
		__host__ __device__ GPUBitBoard &operator=(GPUBitBoard const &src);

		__host__ GPUBitBoard(BitBoard const &src);
		__host__ GPUBitBoard &operator=(BitBoard const &src);

		__host__ operator BitBoard() const;

		__host__ __device__ static board_type GetBlackMoves(GPUBitBoard const &b);
		__host__ __device__ static board_type GetWhiteMoves(GPUBitBoard const &b);

		__host__ __device__ static board_type GetBlackJumps(GPUBitBoard const &b);
		__host__ __device__ static board_type GetWhiteJumps(GPUBitBoard const &b);

		__host__ __device__ static count_type GetBlackPieceCount(GPUBitBoard const &b);
		__host__ __device__ static count_type GetWhitePieceCount(GPUBitBoard const &b);

		__host__ __device__ static count_type GetBlackKingsCount(GPUBitBoard const &b);
		__host__ __device__ static count_type GetWhiteKingsCount(GPUBitBoard const &b);

		__host__ __device__ static void GenMoreWhiteJumps(GPUBitBoard::board_type cell, GPUBitBoard const &board, GPUBitBoard *frontier, int & frontier_size);
		__device__ void static GenMoreWhiteJumpsAtomic(GPUBitBoard::board_type cell, GPUBitBoard const & board, GPUBitBoard * frontier, int * frontier_size);

		__host__ __device__ static void GenMoreBlackJumps(GPUBitBoard::board_type cell, GPUBitBoard const & board, GPUBitBoard *frontier, int & frontier_size);
		__device__ void static GenMoreBlackJumpsAtomic(GPUBitBoard::board_type cell, GPUBitBoard const & board, GPUBitBoard * frontier, int * frontier_size);

		__host__ __device__ static void GPUBitBoard::GenWhiteMove(GPUBitBoard::board_type cell, GPUBitBoard const &board, GPUBitBoard *frontier, int & frontier_size);
		__device__ void static GenWhiteMoveAtomic(GPUBitBoard::board_type cell, GPUBitBoard const & board, GPUBitBoard * frontier, int * frontier_size);
		
		__host__ __device__ static void GenWhiteJump(GPUBitBoard::board_type cell, GPUBitBoard const &board, GPUBitBoard *frontier, int & frontier_size);
		__device__ void static GenWhiteJumpAtomic(GPUBitBoard::board_type cell, GPUBitBoard const & board, GPUBitBoard * frontier, int * frontier_size);

		__host__ __device__ static void GenBlackMove(GPUBitBoard::board_type cell, GPUBitBoard const &board, GPUBitBoard *frontier, int & frontier_size);
		__device__ void static GenBlackMoveAtomic(GPUBitBoard::board_type cell, GPUBitBoard const & board, GPUBitBoard * frontier, int * frontier_size);

		__host__ __device__ static void GenBlackJump(GPUBitBoard::board_type cell, GPUBitBoard const &board, GPUBitBoard *frontier, int & frontier_size);
		__device__ void static GenBlackJumpAtomic(GPUBitBoard::board_type cell, GPUBitBoard const & board, GPUBitBoard * frontier, int * frontier_size);
	};
}