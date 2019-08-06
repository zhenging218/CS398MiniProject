#pragma once

#include "precomp.cuh"

namespace Checkers
{
	namespace
	{
		constexpr GPUBitBoard::board_type EvaluationMask = 0x81188118u;
		constexpr GPUBitBoard::count_type MaxUtility = 10000;
		constexpr GPUBitBoard::count_type MinUtility = -10000;

		constexpr GPUBitBoard::board_type Row0 = 0x10000000u;

		constexpr GPUBitBoard::board_type L3Mask = BitBoard::L3Mask;
		constexpr GPUBitBoard::board_type L5Mask = BitBoard::L5Mask;
		constexpr GPUBitBoard::board_type R3Mask = BitBoard::R3Mask;
		constexpr GPUBitBoard::board_type R5Mask = BitBoard::R5Mask;

		constexpr GPUBitBoard::board_type OddRows = BitBoard::OddRows;

		constexpr GPUBitBoard::board_type BlackKingMask = BitBoard::BlackKingMask;
		constexpr GPUBitBoard::board_type WhiteKingMask = BitBoard::WhiteKingMask;
	}

	__host__ __device__ GPUBitBoard::GPUBitBoard() : white(0xFFF00000u), black(0x00000FFFu), kings(0u)
	{

	}

	__host__ __device__ GPUBitBoard::GPUBitBoard(board_type w, board_type b, board_type k) : white(w), black(b), kings(k)
	{

	}

	__host__ __device__ GPUBitBoard &GPUBitBoard::operator=(GPUBitBoard const &src)
	{
		white = src.white;
		black = src.black;
		kings = src.kings;
		return *this;
	}

	__host__ GPUBitBoard::GPUBitBoard(BitBoard const &src) : white(src.white), black(src.black), kings(src.kings)
	{

	}

	__host__ GPUBitBoard &GPUBitBoard::operator=(BitBoard const &src)
	{
		white = src.white;
		black = src.black;
		kings = src.kings;
		return *this;
	}

	__host__ GPUBitBoard::operator BitBoard() const
	{
		return BitBoard(white, black, kings);
	}

	__device__ GPUBitBoard::board_type GPUBitBoard::GetBlackMoves(GPUBitBoard const &b)
	{
		const board_type not_occupied = ~(b.black | b.white);
		const board_type black_kings = b.black & b.kings;
		board_type LL = (((not_occupied & R5Mask) >> 5) & b.black) | ((not_occupied >> 4) & b.black);
		board_type LR = (((not_occupied) >> 4) & b.black) | (((not_occupied & R3Mask) >> 3) & b.black);
		board_type UL = (((not_occupied & L5Mask) << 5) & black_kings) | ((not_occupied << 4) & black_kings);
		board_type UR = (((not_occupied << 4) & black_kings) | ((not_occupied & L3Mask) << 3) & black_kings);

		return  (LL | LR | UL | UR);
	}

	__device__ GPUBitBoard::board_type GPUBitBoard::GetWhiteMoves(GPUBitBoard const &b)
	{
		const board_type not_occupied = ~(b.black | b.white);
		const board_type white_kings = b.white & b.kings;
		board_type LL = (((not_occupied & R5Mask) >> 5) & white_kings) | ((not_occupied >> 4) & white_kings);
		board_type LR = (((not_occupied) >> 4) & white_kings) | (((not_occupied & R3Mask) >> 3) & white_kings);
		board_type UL = (((not_occupied & L5Mask) << 5) & b.white) | ((not_occupied << 4) & b.white);
		board_type UR = (((not_occupied << 4) & b.white) | ((not_occupied & L3Mask) << 3) & b.white);

		return  (LL | LR | UL | UR);
	}

	__device__ GPUBitBoard::board_type GPUBitBoard::GetBlackJumps(GPUBitBoard const &b)
	{
		const board_type not_occupied = ~(b.black | b.white);
		const board_type black_kings = b.black & b.kings;

		board_type LL_even_to_odd = (((((not_occupied & R5Mask) >> 5) & b.white) >> 4) & b.black);
		board_type LR_even_to_odd = (((((not_occupied >> 4) & b.white) & R3Mask) >> 3) & b.black);
		board_type UL_even_to_odd = (((((not_occupied & L5Mask) << 5) & b.white) << 4) & black_kings);
		board_type UR_even_to_odd = (((((not_occupied << 4) & b.white) & L3Mask) << 3) & black_kings);

		board_type LL_odd_to_even = (((((not_occupied >> 4) & b.white) & R5Mask) >> 5) & b.black);
		board_type LR_odd_to_even = (((((not_occupied & R3Mask) >> 3) & b.white) >> 4) & b.black);
		board_type UL_odd_to_even = (((((not_occupied << 4) & b.white) & L5Mask) << 5) & black_kings);
		board_type UR_odd_to_even = (((((not_occupied & L3Mask) << 3) & b.white) << 4) & black_kings);

		board_type move = (LL_even_to_odd | LR_even_to_odd | UL_even_to_odd | UR_even_to_odd |
			LL_odd_to_even | LR_odd_to_even | UL_odd_to_even | UR_odd_to_even);
		return move;
	}

	__device__ GPUBitBoard::board_type GPUBitBoard::GetWhiteJumps(GPUBitBoard const &b)
	{
		const board_type not_occupied = ~(b.black | b.white);
		const board_type white_kings = b.white & b.kings;

		board_type LL_even_to_odd = (((((not_occupied & R5Mask) >> 5) & b.black) >> 4) & white_kings);
		board_type LR_even_to_odd = (((((not_occupied >> 4) & b.black) & R3Mask) >> 3) & white_kings);
		board_type UL_even_to_odd = (((((not_occupied & L5Mask) << 5) & b.black) << 4) & b.white);
		board_type UR_even_to_odd = (((((not_occupied << 4) & b.black) & L3Mask) << 3) & b.white);

		board_type LL_odd_to_even = (((((not_occupied >> 4) & b.black) & R5Mask) >> 5) & white_kings);
		board_type LR_odd_to_even = (((((not_occupied & R3Mask) >> 3) & b.black) >> 4) & white_kings);
		board_type UL_odd_to_even = (((((not_occupied << 4) & b.black) & L5Mask) << 5) & b.white);
		board_type UR_odd_to_even = (((((not_occupied & L3Mask) << 3) & b.black) << 4) & b.white);

		board_type move = (LL_even_to_odd | LR_even_to_odd | UL_even_to_odd | UR_even_to_odd |
			LL_odd_to_even | LR_odd_to_even | UL_odd_to_even | UR_odd_to_even);
		return move;
	}

	__device__ GPUBitBoard::count_type GPUBitBoard::GetBlackPieceCount(GPUBitBoard const &b)
	{
		return GPUSWAR32(b.black);
	}

	__device__ GPUBitBoard::count_type GPUBitBoard::GetWhitePieceCount(GPUBitBoard const &b)
	{
		return GPUSWAR32(b.white);
	}

	__device__ GPUBitBoard::count_type GPUBitBoard::GetBlackKingsCount(GPUBitBoard const &b)
	{
		return GPUSWAR32(b.black & b.kings);
	}

	__device__ GPUBitBoard::count_type GPUBitBoard::GetWhiteKingsCount(GPUBitBoard const &b)
	{
		return GPUSWAR32(b.white & b.kings);
	}
}