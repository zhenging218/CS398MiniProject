#include "precomp.h"
#include <utility>
#include <array>

// logic referenced from http://www.3dkingdoms.com/checkers/bitboards.htm

namespace Checkers
{
	namespace
	{
		constexpr BitBoard::board_type L3Mask = 0xF0707070u;
		constexpr BitBoard::board_type L5Mask = 0x0D0D0D00u;
		constexpr BitBoard::board_type R3Mask = 0x0D0D0D0Du;
		constexpr BitBoard::board_type R5Mask = 0x00707070u;

		constexpr BitBoard::board_type OddRows = 0xF0F0F0F0u;

		constexpr BitBoard::board_type BlackKingMask = 0xF000000;
		constexpr BitBoard::board_type WhiteKingMask = 0x000000F;

		BitBoard::board_type InitializeWhitePieces()
		{
			return 0xFFF00000;
		}

		BitBoard::board_type InitializeBlackPieces()
		{
			return 0x00000FFF;
		}
	}

	BitBoard::BitBoard() : white(InitializeWhitePieces()), black(InitializeBlackPieces()), kings(0)
	{

	}

	BitBoard::BitBoard(BitBoard::board_type w, BitBoard::board_type b, BitBoard::board_type k) : white(w), black(b), kings(k)
	{

	}


	BitBoard::board_type BitBoard::GetWhiteMoves() const
	{
		const board_type not_occupied = ~(black | white);

		board_type moves = not_occupied << 4;
		moves |= ((not_occupied & L3Mask) << 3);
		moves |= ((not_occupied & L5Mask) << 5);
		moves &= white;

		const board_type white_kings = white & kings;

		if (white_kings)
		{
			moves |= ((not_occupied >> 4) & white_kings);
			moves |= (((not_occupied & R3Mask) >> 3) & white_kings);
			moves |= (((not_occupied & R5Mask) >> 5) & white_kings);
		}

		return moves;
	}

	BitBoard::board_type BitBoard::GetBlackMoves() const
	{
		const board_type not_occupied = ~(black | white);

		board_type moves = not_occupied << 4;
		moves |= ((not_occupied & L3Mask) << 3);
		moves |= ((not_occupied & L5Mask) << 5);
		moves &= black;

		const board_type black_kings = black & kings;

		if (black_kings)
		{
			moves |= ((not_occupied >> 4) & black_kings);
			moves |= (((not_occupied & R3Mask) >> 3) & black_kings);
			moves |= (((not_occupied & R5Mask) >> 5) & black_kings);
		}

		return moves;
	}

	BitBoard::board_type BitBoard::GetWhiteJumps() const
	{
		const board_type not_occupied = ~(black | white);
		board_type jumps = 0;

		board_type non_king_jumps = (not_occupied << 4) & black;
		if (non_king_jumps)
		{
			jumps |= ((((non_king_jumps & L3Mask) << 3) | ((non_king_jumps & L5Mask) << 5)) & white);
		}
		non_king_jumps = ((((not_occupied & L3Mask) << 3) | ((not_occupied & L5Mask) << 5)) & black);
		jumps |= ((non_king_jumps << 4) & white);

		const board_type white_kings = white & kings;
		if (white_kings)
		{
			board_type king_jumps = (not_occupied >> 4) & black;
			if (king_jumps)
			{
				jumps |= ((((king_jumps & R3Mask) >> 3) | ((king_jumps & R5Mask) >> 5)) & white_kings);
			}
			king_jumps = ((((not_occupied & R3Mask) >> 3) | ((not_occupied & R5Mask) >> 5)) & black);
			jumps |= ((king_jumps >> 4) & white_kings);
		}

		return jumps;
	}

	BitBoard::board_type BitBoard::GetBlackJumps() const
	{
		const board_type not_occupied = ~(black | white);
		board_type jumps = 0;

		board_type non_king_jumps = (not_occupied >> 4) & white;
		if (non_king_jumps)
		{
			jumps |= ((((non_king_jumps & R3Mask) >> 3) | ((non_king_jumps & R5Mask) >> 5)) & black);
		}
		non_king_jumps = ((((not_occupied & R3Mask) >> 3) | ((not_occupied & R5Mask) >> 5)) & white);
		jumps |= ((non_king_jumps >> 4) & black);

		const board_type black_kings = black & kings;
		if (black_kings)
		{
			board_type king_jumps = (not_occupied << 4) & white;
			if (king_jumps)
			{
				jumps |= ((((king_jumps & L3Mask) << 3) | ((king_jumps & L5Mask) << 5)) & black_kings);
			}
			king_jumps = ((((not_occupied & L3Mask) << 3) | ((not_occupied & L5Mask) << 5)) & white);
			jumps |= ((king_jumps << 4) & black_kings);
		}

		return jumps;
	}

	std::ostream &operator<<(std::ostream &os, BitBoard const &src)
	{
		using board_type = BitBoard::board_type;

		board_type white = src.white;
		board_type black = src.black;
		board_type white_kings = src.white & src.kings;
		board_type black_kings = src.black & src.kings;

		int start = 0;
		int x = 1;

		for (int j = 0; j < 8; ++j)
		{
			os << "|";
			for (int i = 0; i < 8; ++i)
			{
				// check if index is even or odd
				if (start)
				{
					// odd
					if (i % 2)
					{
						os << ((x & white) ? (x & (white_kings) ? "WW" : " W") : ((x & black) ? (x & black_kings) ? "BB" : " B" : "  ")) << "|";
						x <<= 1;
					}
					else
					{
						os << "  |";
					}
				}
				else
				{
					// even
					if (!(i % 2))
					{
						os << ((x & white) ? (x & (white_kings) ? "WW" : " W") : ((x & black) ? (x & black_kings) ? "BB" : " B" : "  ")) << "|";
						x <<= 1;
					}
					else
					{
						os << "  |";
					}
				}
			}
			os << "\n";
			start = 1 - start;
		}

		return os;
	}

	void BitBoard::GetPossibleBlackMoves(Move *dst) const
	{
		board_type moves = GetBlackMoves();
		board_type jumps = GetBlackJumps();
		board_type empty = ~(white | black);

		int k = 0;
		board_type i = 1;

		while (moves && i && k < 48)
		{

			if (moves & i)
			{
				if (OddRows & i)
				{
					// odd rows
					if (((i & kings) << 4) & empty) // UL
					{
						dst[k] = Move(BitBoard(white, (black & ~i) | (i << 4), (kings & ~i) | (((kings & i) << 4)|((i << 4) & BlackKingMask))) , false);
						++k;
					}

					if (((i & kings) << 3) & empty && k < 48) // UR
					{
						dst[k] = Move(BitBoard(white, (black & ~i) | (i << 3), (kings & ~i) | (((kings & i) << 3) | ((i << 3) & BlackKingMask))), false);
						++k;
					}

					if ((i >> 3) & empty && k < 48) // LR
					{
						dst[k] = Move(BitBoard(white, (black & ~i) | (i >> 3), (kings & ~i) | (((kings & i) >> 3) | ((i >> 3) & BlackKingMask))), false);
						++k;
					}

					if ((i >> 4) & empty && k < 48) // LL
					{
						dst[k] = Move(BitBoard(white, (black & ~i) | (i >> 4), (kings & ~i) | (((kings & i) >> 4) | ((i >> 4) & BlackKingMask))), false);
						++k;
					}

				}
				else
				{
					// even rows
					if (((i & kings) << 5) & empty) // UL
					{
						dst[k] = Move(BitBoard(white, (black & ~i) | (i << 5), (kings & ~i) | (((kings & i) << 5) | ((i << 5) & BlackKingMask))), false);
						++k;
					}

					if (((i & kings) << 4) & empty && k < 48) // UR
					{
						dst[k] = Move(BitBoard(white, (black & ~i) | (i << 4), (kings & ~i) | (((kings & i) << 4) | ((i << 4) & BlackKingMask))), false);
						++k;
					}

					if ((i >> 4) & empty && empty < 48) // LR
					{
						dst[k] = Move(BitBoard(white, (black & ~i) | (i >> 4), (kings & ~i) | (((kings & i) >> 4) | ((i >> 4) & BlackKingMask))), false);
						++k;
					}

					if ((i >> 5) & empty && empty < 48) // LL
					{
						dst[k] = Move(BitBoard(white, (black & ~i) | (i >> 5), (kings & ~i) | (((kings & i) >> 5) | ((i >> 5) & BlackKingMask))), false);
						++k;
					}
				}
			}

			moves &= ~i;
			i << 1;
		}

		i = 1;

		while (jumps && i && k < 48)
		{
			if (jumps & i)
			{
				if (OddRows & i)
				{
					// odd rows, jump lands in odd row (enemy piece in even row)
					if (((i & kings) << 4) & white) // UL from odd
					{
						board_type j = i << 4;
						if ((j << 5) & empty) // UL from even
						{
							dst[k] = Move(BitBoard((white & ~j), (black & ~i) | (j << 5), (kings & ~i) | ((((kings & i) << 4) << 5) | ((j << 5) & BlackKingMask))), true);
			
							++k;
						}
					}

					if (((i & kings) << 3) & white && k < 48) // UR
					{
						board_type j = i << 3;
						if ((j << 4) & empty) // UR from even
						{
							dst[k] = Move(BitBoard((white & ~j), (black & ~i) | (j << 4), (kings & ~i) | ((((kings & i) << 3) << 4) | ((j << 4) & BlackKingMask))), true);
							++k;
						}
					}

					if ((i >> 3) & white && k < 48) // LR
					{
						board_type j = i >> 3;
						if ((j >> 4) & empty)// LR from even
						{
							dst[k] = Move(BitBoard((white & ~j), (black & ~i) | (j >> 4), (kings & ~i) | ((((kings & i) >> 3)>>4) | ((j >> 4) & BlackKingMask))), true);
							++k;
						}
					}

					if ((i >> 4) & white && k < 48) // LL
					{
						board_type j = i >> 4;
						if ((j >> 5) & empty) // LL from even
						{
							dst[k] = Move(BitBoard((white & ~j), (black & ~i) | (j >> 5), (kings & ~i) | ((((kings & i) >> 4) >> 5) | ((j >> 5) & BlackKingMask))), true);
							++k;
						}
					}

				}
				else
				{
					// even rows
					if (((i & kings) << 5) & white) // UL from even
					{
						board_type j = i << 4;
						if ((j << 4) & empty) // UL from odd
						{
							dst[k] = Move(BitBoard((white & ~j), (black & ~i) | (j << 4), (kings & ~i) | ((((kings & i) << 5) << 4) | ((j << 4) & BlackKingMask))), true);
							++k;
						}
					}

					if (((i & kings) << 4) & white && k < 48) // UR from even
					{
						board_type j = i << 3;
						if ((j << 3) & empty) // UR from odd
						{
							dst[k] = Move(BitBoard((white & ~j), (black & ~i) | (j << 4), (kings & ~i) | ((((kings & i) << 3) << 4) | ((j << 4) & BlackKingMask))), true);
							++k;
						}
					}

					if ((i >> 3) & white && k < 48) // LR
					{
						board_type j = i >> 3;
						if ((j >> 4) & empty)// LR from even
						{
							dst[k] = Move(BitBoard((white & ~j), (black & ~i) | (j >> 4), (kings & ~i) | ((((kings & i) >> 3) >> 4) | ((j >> 4) & BlackKingMask))), true);
							++k;
						}
					}

					if ((i >> 4) & white && k < 48) // LL
					{
						board_type j = i >> 4;
						if ((j >> 5) & empty) // LL from even
						{
							dst[k] = Move(BitBoard((white & ~j), (black & ~i) | (j >> 5), (kings & ~i) | ((((kings & i) >> 4) >> 5) | ((j >> 5) & BlackKingMask))), true);
							++k;
						}
					}
				}
			}

			jumps &= ~i;
			i << 1;
		}
	}
}