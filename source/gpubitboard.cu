#pragma once

#include "gpubitboard.cuh"

namespace Checkers
{
	__host__ __device__ GPUBitBoard::GPUBitBoard() : white(0xFFF00000u), black(0x00000FFFu), kings(0u), valid(false)
	{

	}

	__host__ __device__ GPUBitBoard::GPUBitBoard(board_type w, board_type b, board_type k, bool v) : white(w), black(b), kings(k), valid(v)
	{

	}

	__host__ __device__ GPUBitBoard &GPUBitBoard::operator=(GPUBitBoard const &src)
	{
		white = src.white;
		black = src.black;
		kings = src.kings;
		valid = src.valid;
		return *this;
	}

	__host__ GPUBitBoard::GPUBitBoard(BitBoard const &src) : white(src.white), black(src.black), kings(src.kings), valid(false)
	{

	}

	__host__ GPUBitBoard &GPUBitBoard::operator=(BitBoard const &src)
	{
		white = src.white;
		black = src.black;
		kings = src.kings;
		valid = false;
		return *this;
	}

	__host__ GPUBitBoard::operator BitBoard() const
	{
		return BitBoard(white, black, kings);
	}
}