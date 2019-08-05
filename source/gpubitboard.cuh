#pragma once

#include "bitboard.h"
#include "minimax.h"
#include <thrust/device_vector.h>

namespace Checkers
{
	struct GPUBitBoard
	{
		using board_type = BitBoard::board_type;
		using count_type = BitBoard::count_type;
		
		static constexpr board_type L3Mask = BitBoard::L3Mask;
		static constexpr board_type L5Mask = BitBoard::L5Mask;
		static constexpr board_type R3Mask = BitBoard::R3Mask;
		static constexpr board_type R5Mask = BitBoard::R5Mask;

		static constexpr board_type OddRows = BitBoard::OddRows;

		static constexpr board_type BlackKingMask = BitBoard::BlackKingMask;
		static constexpr board_type WhiteKingMask = BitBoard::WhiteKingMask;

		board_type white, black, kings;

		__host__ __device__ GPUBitBoard();
		__host__ __device__ GPUBitBoard(board_type w, board_type b, board_type k);
		__host__ __device__ GPUBitBoard &operator=(GPUBitBoard const &src);

		__host__ GPUBitBoard(BitBoard const &src);
		__host__ GPUBitBoard &operator=(BitBoard const &src);

		__host__ operator BitBoard() const;

		__device__ static bool GetPossibleWhiteMoves(GPUBitBoard const &b, thrust::device_vector<GPUBitBoard> &dst);
		__device__ static bool GetPossibleBlackMoves(GPUBitBoard const &b, thrust::device_vector<GPUBitBoard> &dst);

		__device__ static board_type GetBlackMoves(BitBoard const &b);
		__device__ static board_type GetWhiteMoves(BitBoard const &b);

		__device__ static board_type GetBlackJumps(BitBoard const &b);
		__device__ static board_type GetWhiteJumps(BitBoard const &b);

		__device__ static count_type GetBlackPieceCount(BitBoard const &b);
		__device__ static count_type GetWhitePieceCount(BitBoard const &b);

		__device__ static count_type GetBlackKingsCount(BitBoard const &b);
		__device__ static count_type GetWhiteKingsCount(BitBoard const &b);
	};
}