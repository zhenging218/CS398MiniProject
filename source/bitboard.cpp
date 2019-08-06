#include "precomp.h"
#include <utility>
#include <iostream>

// logic referenced from http://www.3dkingdoms.com/checkers/bitboards.htm
// minimax referenced from https://github.com/billjeffries/jsCheckersAI

namespace Checkers
{
	namespace
	{
		constexpr BitBoard::board_type EvaluationMask = 0x81188118u;
		constexpr BitBoard::count_type MaxUtility = 10000;
		constexpr BitBoard::count_type MinUtility = -10000;

		constexpr BitBoard::board_type Row0 = 0x10000000u;

		constexpr BitBoard::board_type InitializeWhitePieces() noexcept
		{
			return 0xFFF00000u;
		}

		constexpr BitBoard::board_type InitializeBlackPieces() noexcept
		{
			return 0x00000FFFu;
		}
	}

	BitBoard::count_type BitBoard::GetBlackPieceCount(BitBoard const &b)
	{
		return SWAR32(b.black);
	}

	BitBoard::count_type BitBoard::GetWhitePieceCount(BitBoard const &b)
	{
		return SWAR32(b.white);
	}

	BitBoard::count_type BitBoard::GetBlackKingsCount(BitBoard const &b)
	{
		return SWAR32(b.black & b.kings);
	}

	BitBoard::count_type BitBoard::GetWhiteKingsCount(BitBoard const &b)
	{
		return SWAR32(b.white & b.kings);
	}

	BitBoard::BitBoard() : white(InitializeWhitePieces()), black(InitializeBlackPieces()), kings(0)
	{

	}

	BitBoard::BitBoard(BitBoard::board_type w, BitBoard::board_type b, BitBoard::board_type k) : white(w), black(b), kings(k)
	{

	}


	BitBoard::board_type BitBoard::GetWhiteMoves(BitBoard const &b)
	{
		const board_type not_occupied = ~(b.black | b.white);
		const board_type white_kings = b.white & b.kings;
		board_type LL = (((not_occupied & R5Mask) >> 5) & white_kings) | ((not_occupied >> 4) & white_kings);
		board_type LR = (((not_occupied) >> 4) & white_kings) | (((not_occupied & R3Mask) >> 3) & white_kings);
		board_type UL = (((not_occupied & L5Mask) << 5) & b.white) | ((not_occupied << 4) & b.white);
		board_type UR = (((not_occupied << 4) & b.white) | ((not_occupied & L3Mask) << 3) & b.white);

		return  (LL | LR | UL | UR);
	}

	BitBoard::board_type BitBoard::GetBlackMoves(BitBoard const &b)
	{
		const board_type not_occupied = ~(b.black | b.white);
		const board_type black_kings = b.black & b.kings;
		board_type LL = (((not_occupied & R5Mask) >> 5) & b.black) | ((not_occupied >> 4) & b.black);
		board_type LR = (((not_occupied) >> 4) & b.black) | (((not_occupied & R3Mask) >> 3) & b.black);
		board_type UL = (((not_occupied & L5Mask) << 5) & black_kings) | ((not_occupied << 4) & black_kings);
		board_type UR = (((not_occupied << 4) & black_kings) | ((not_occupied & L3Mask) << 3) & black_kings);

		return  (LL | LR | UL | UR);
	}

	BitBoard::board_type BitBoard::GetWhiteJumps(BitBoard const &b)
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

	BitBoard::board_type BitBoard::GetBlackJumps(BitBoard const &b)
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

	std::ostream &operator<<(std::ostream &os, BitBoard const &src)
	{
		using board_type = BitBoard::board_type;

		board_type white = src.white;
		board_type black = src.black;
		board_type white_kings = src.white & src.kings;
		board_type black_kings = src.black & src.kings;

		for (BitBoard::board_type i = Row0; i != 0; i = i >> 4)
		{
			os << "|";
			BitBoard::board_type j = i;
			if (i & BitBoard::OddRows)
			{
				for (int k = 0; k < 8; ++k)
				{
					if (k % 2) // odd
					{
						if (j & white)
						{
							os << (j & white_kings ? "WW|" : " W|");
						}
						else if (j & black)
						{
							os << (j & black_kings ? "BB|" : " B|");
						}
						else
						{
							os << "  |";
						}
						j = j << 1;
					}
					else // even
					{
						os << "  |";
					}
				}
			}
			else
			{
				for (int k = 0; k < 8; ++k)
				{
					if (!(k % 2)) // even
					{
						if (j & white)
						{
							os << (j & white_kings ? "WW|" : " W|");
						}
						else if (j & black)
						{
							os << (j & black_kings ? "BB|" : " B|");
						}
						else
						{
							os << "  |";
						}
						j = j << 1;
					}
					else // odd
					{
						os << "  |";
					}
				}
			}
			os << ((i >> 4) ? "\n" : "");
		}

		return os;
	}
}