#pragma once

#include "gpubitboard.cuh"
#include "precomp.cuh"

namespace Checkers
{
	namespace
	{
		constexpr BitBoard::board_type EvaluationMask = 0x81188118u;
		constexpr BitBoard::count_type MaxUtility = 10000;
		constexpr BitBoard::count_type MinUtility = -10000;

		constexpr BitBoard::board_type Row0 = 0x10000000u;
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
}