#pragma once

#include "gpubitboard.cuh"
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

	__device__ bool GPUBitBoard::GetPossibleWhiteMoves(GPUBitBoard const &b, thrust::device_vector<GPUBitBoard> &dst)
	{
		board_type moves = GetWhiteMoves(b);
		board_type jumps = GetWhiteJumps(b);
		board_type empty = ~(b.white | b.black);

		board_type i = 1;
		bool jumped = false;

		if (!jumps)
		{
			while (moves && i)
			{

				if (moves & i)
				{

					if (OddRows & i) // odd rows
					{
						if (((i & b.kings) << 4) & empty)//UL
						{
							dst.push_back(BitBoard((b.white & ~i) | (i << 4), b.black, (b.kings & ~i) | (i << 4)));

						}
						if (((((i & b.kings))& L5Mask) << 5) & empty)//UR
						{
							dst.push_back(BitBoard((b.white & ~i) | (i << 5), b.black, (b.kings & ~i) | (i << 5)));

						}

						if ((i >> 4) & empty)//LL
						{
							dst.push_back(BitBoard((b.white & ~i) | (i >> 4), b.black, (b.kings & ~i) | (((b.kings & i) >> 4) | ((i >> 4) & WhiteKingMask))));

						}
						if (((i & R3Mask) >> 3) & empty)//LR
						{
							dst.push_back(BitBoard((b.white & ~i) | (i >> 3), b.black, (b.kings & ~i) | (((b.kings & i) >> 3) | ((i >> 3) & WhiteKingMask))));

						}

					}
					else // even rows
					{
						if ((((i & L3Mask) & b.kings) << 3) & empty) //UL
						{
							dst.push_back(BitBoard((b.white & ~i) | (i << 3), b.black, (b.kings & ~i) | (i << 3)));

						}
						if (((i & b.kings) << 4) & empty) //UR
						{
							dst.push_back(BitBoard((b.white & ~i) | (i << 4), b.black, (b.kings & ~i) | (i << 4)));

						}

						if (((i & R5Mask) >> 5) & empty) //LL
						{
							dst.push_back(BitBoard((b.white & ~i) | (i >> 5), b.black, (b.kings & ~i) | (((b.kings & i) >> 5) | ((i >> 5) & WhiteKingMask))));
						}
						if ((i >> 4) & empty)//LR
						{
							dst.push_back(BitBoard((b.white & ~i) | (i >> 4), b.black, (b.kings & ~i) | (((b.kings & i) >> 4) | ((i >> 4) & WhiteKingMask))));
						}
					}
				}

				moves &= ~i;
				i = i << 1;
			}
		}
		else
		{
			jumped = true;
			while (jumps && i)
			{

				if (jumps & i)
				{
					if (OddRows & i) // odd rows
					{
						if (((i & b.kings) << 4) & b.black) //UL from odd
						{
							board_type j = i << 4;
							if (((j & L3Mask) << 3) & empty) // UL from even
							{
								dst.push_back(BitBoard((b.white & ~i) | (j << 3), (b.black & ~j), (b.kings & (~i ^ j)) | (j << 3)));
							}
						}

						if ((((i & L5Mask)& b.kings) << 5) & b.black) //UR from odd
						{
							board_type j = i << 5;
							if ((j << 4) & empty) // UR from even
							{
								dst.push_back(BitBoard((b.white & ~i) | (j << 4), (b.black & ~j), (b.kings & (~i ^ j)) | (j << 4)));
							}
						}

						if ((i >> 4) & b.black) //LL from odd
						{
							board_type j = i >> 4;
							if (((j & R5Mask) >> 5) & empty) // LL from even
							{
								dst.push_back(BitBoard((b.white & ~i) | (j >> 5), (b.black & ~j), (b.kings & (~i ^ j)) | ((((b.kings & i) >> 4) >> 5) | ((j >> 5) & WhiteKingMask))));
							}
						}

						if (((i & R3Mask) >> 3) & b.black)//LR
						{
							board_type j = i >> 3;
							if ((j >> 4) & empty)
							{
								dst.push_back(BitBoard((b.white & ~i) | (j >> 4), (b.black & ~j), (b.kings & (~i ^ j)) | ((((b.kings & i) >> 3) >> 4) | ((j >> 4) & WhiteKingMask))));
							}
						}
					}
					else // even rows
					{
						if ((((i & L3Mask) & b.kings) << 3) & b.black) //UL
						{
							board_type j = i << 3;
							if ((j << 4) & empty)
							{
								dst.push_back(BitBoard((b.white & ~i) | (j << 4), (b.black & ~j), (b.kings & (~i ^ j)) | (j << 4)));
							}


						}
						if (((i & b.kings) << 4) & b.black) //UR
						{
							board_type j = i << 4;
							if (((j & L5Mask) << 5) & empty)
							{
								dst.push_back(BitBoard((b.white & ~i) | (j << 5), (b.black & ~j), (b.kings & (~i ^ j)) | (j << 5)));
							}

						}

						if (((i & R5Mask) >> 5) & b.black) //LL
						{
							board_type j = i >> 5;
							if ((j >> 4) & empty)
							{
								dst.push_back(BitBoard((b.white & ~i) | (j >> 4), (b.black & ~j), (b.kings & (~i ^ j)) | ((((b.kings & i) >> 5) >> 4) | ((j >> 4) & WhiteKingMask))));
							}
						}
						if ((i >> 4) & b.black)//LR
						{
							board_type j = i >> 4;
							if (((j & R3Mask) >> 3) & empty)
							{
								dst.push_back(BitBoard((b.white & ~i) | (j >> 3), (b.black & ~j), (b.kings & (~i ^ j)) | ((((b.kings & i) >> 4) >> 3) | ((j >> 3) & WhiteKingMask))));
							}

						}
					}
				}

				jumps &= ~i;
				i = i << 1;
			}
			return true;
		}
		return false;
	}

	__device__ bool GPUBitBoard::GetPossibleBlackMoves(GPUBitBoard const &b, thrust::device_vector<GPUBitBoard> &dst)
	{
		board_type moves = GetBlackMoves(b);
		board_type jumps = GetBlackJumps(b);
		board_type empty = ~(b.black | b.white);

		board_type i = 1;
		bool jumped = false;

		if (!jumps)
		{
			while (moves && i)
			{

				if (moves & i)
				{
					if (OddRows & i) // odd rows
					{
						if ((i << 4) & empty)//UL
						{
							dst.push_back(BitBoard(b.white, (b.black & ~i) | (i << 4), (b.kings & ~i) | (((b.kings & i) << 4) | ((i << 4) & BlackKingMask))));

						}
						if (((i & L5Mask) << 5) & empty)//UR
						{

							dst.push_back(BitBoard(b.white, (b.black & ~i) | (i << 5), (b.kings & ~i) | (((b.kings & i) << 5) | ((i << 5) & BlackKingMask))));

						}

						if (((i & b.kings) >> 4) & empty)//LL
						{
							dst.push_back(BitBoard(b.white, (b.black & ~i) | (i >> 4), (b.kings & ~i) | (i >> 4)));

						}
						if ((((i&R3Mask) & b.kings) >> 3) & empty)//LR
						{
							dst.push_back(BitBoard(b.white, (b.black & ~i) | (i >> 3), (b.kings & ~i) | (i >> 3)));

						}
					}
					else // even rows
					{
						if (((i & L3Mask) << 3) & empty)//UL
						{
							//

							dst.push_back(BitBoard(b.white, (b.black & ~i) | (i << 3), (b.kings & ~i) | (((b.kings & i) << 3) | ((i << 3) & BlackKingMask))));

						}
						if ((i << 4) & empty)//UR
						{

							//
							dst.push_back(BitBoard(b.white, (b.black & ~i) | (i << 4), (b.kings & ~i) | (((b.kings & i) << 4) | ((i << 4) & BlackKingMask))));

						}

						if ((((i  & b.kings)& R5Mask) >> 5) & empty)//LL
						{
							dst.push_back(BitBoard(b.white, (b.black & ~i) | (i >> 5), (b.kings & ~i) | (i >> 5)));


						}
						if (((i  & b.kings) >> 4) & empty)//LR
						{
							dst.push_back(BitBoard(b.white, (b.black & ~i) | (i >> 4), (b.kings & ~i) | (i >> 4)));


						}

					}
				}

				moves &= ~i;
				i = i << 1;
			}
		}
		else
		{
			jumped = true;
			while (jumps && i)
			{
				if (jumps & i)
				{
					if (OddRows & i)
					{
						// odd rows, jump lands in odd row (enemy piece in even row)
						if ((i << 4) & b.white) // UL from odd
						{
							board_type j = i << 4;
							if (((j & L3Mask) << 3) & empty) // UL from even
							{
								dst.push_back(BitBoard((b.white & ~j), (b.black & ~i) | (j << 3), (b.kings & (~i ^ j)) | ((((b.kings & i) << 4) << 3) | ((j << 3) & BlackKingMask))));
							}
						}

						if (((i & L5Mask) << 5) & b.white) // UR from odd
						{
							board_type j = i << 5;
							if ((j << 4) & empty) // UR from even
							{
								dst.push_back(BitBoard((b.white & ~j), (b.black & ~i) | (j << 4), (b.kings & (~i ^ j)) | ((((b.kings & i) << 5) << 4) | ((j << 4) & BlackKingMask))));
							}
						}

						if ((((i & R3Mask) & b.kings) >> 3) & b.white) // LR from odd
						{
							board_type j = i >> 3;
							if ((j >> 4) & empty)// LR from even
							{
								dst.push_back(BitBoard((b.white & ~j), (b.black & ~i) | (j >> 4), (b.kings & (~i ^ j)) | (j >> 4)));
							}
						}

						if (((i & b.kings) >> 4) & b.white) // LL from odd
						{
							board_type j = i >> 4;
							if (((j & R5Mask) >> 5) & empty) // LL from even
							{
								dst.push_back(BitBoard((b.white & ~j), (b.black & ~i) | (j >> 5), (b.kings & (~i ^ j)) | (j >> 5)));
							}
						}
					}
					else
					{
						// even rows
						if ((((i & L3Mask)) << 3) & b.white) // UL from even
						{
							board_type j = i << 3;
							if ((j << 4) & empty) // UL from odd
							{
								dst.push_back(BitBoard((b.white & ~j), (b.black & ~i) | (j << 4), (b.kings & (~i ^ j)) | ((((b.kings & i) << 3) << 4) | ((j << 4) & BlackKingMask))));
							}
						}

						if ((i << 4) & b.white) // UR from even
						{
							board_type j = i << 4;
							if (((j & L5Mask) << 5) & empty) // UR from odd
							{
								dst.push_back(BitBoard((b.white & ~j), (b.black & ~i) | (j << 5), (b.kings & (~i ^ j)) | ((((b.kings & i) << 4) << 5) | ((j << 5) & BlackKingMask))));
							}
						}

						if (((i & b.kings) >> 4) & b.white) // LR from even
						{
							board_type j = i >> 4;
							if (((j & R3Mask) >> 3) & empty)// LR from odd
							{
								dst.push_back(BitBoard((b.white & ~j), (b.black & ~i) | (j >> 3), (b.kings & (~i ^ j)) | (j >> 3)));
							}
						}

						if ((((i & b.kings) & R5Mask) >> 5) & b.white) // LL from even
						{
							board_type j = i >> 5;
							if ((j >> 4) & empty) // LL from odd
							{
								dst.push_back(BitBoard((b.white & ~j), (b.black & ~i) | (j >> 4), (b.kings & (~i ^ j)) | (j >> 4)));
							}
						}
					}
				}

				jumps &= ~i;
				i = i << 1;
			}
			return true;
		}
		return false;
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