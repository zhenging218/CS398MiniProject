#include "precomp.h"
#include <utility>
#include <array>
#include <iostream>

// logic referenced from http://www.3dkingdoms.com/checkers/bitboards.htm
// minimax referenced from https://github.com/billjeffries/jsCheckersAI

namespace Checkers
{
	namespace
	{
		constexpr BitBoard::board_type EvaluationMask = 0x81188118u;
		constexpr BitBoard::utility_type MaxUtility = 10000;
		constexpr BitBoard::utility_type MinUtility = -10000;

		constexpr BitBoard::board_type Row0 = 0x10000000u;

		std::uint32_t SWAR(BitBoard::board_type i)
		{
			// SWAR algorithm: count bits
			// https://stackoverflow.com/questions/22081738/how-does-this-algorithm-to-count-the-number-of-set-bits-in-a-32-bit-integer-work

			i = i - ((i >> 1) & 0x55555555);
			i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
			return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
		}

		BitBoard::board_type InitializeWhitePieces()
		{
			return 0xFFF00000;
		}

		BitBoard::board_type InitializeBlackPieces()
		{
			return 0x00000FFF;
		}

		BitBoard::utility_type EvaluatePosition(BitBoard::board_type board)
		{
			BitBoard::board_type corners = board & EvaluationMask;
			// BitBoard::board_type others = ~corners;
			// return SWAR(corners) * 2 + SWAR(others);
			return SWAR(corners);
		}
	}

	Move::Move() : board(), jump(false)
	{

	}

	Move::Move(BitBoard const &bb, bool j) : board(bb), jump(j)
	{

	}

	BitBoard::utility_type BitBoard::GetBlackPieceCount() const
	{
		return SWAR(black);
	}

	BitBoard::utility_type BitBoard::GetWhitePieceCount() const
	{
		return SWAR(white);
	}

	BitBoard::utility_type BitBoard::GetBlackUtility() const
	{
		std::uint32_t white_pieces = SWAR(white);
		std::uint32_t black_pieces = SWAR(black);
		std::uint32_t white_kings = SWAR(white & kings);
		std::uint32_t black_kings = SWAR(black & kings);
		std::uint32_t black_eval = EvaluatePosition(black);
		std::uint32_t white_eval = EvaluatePosition(white);

		std::uint32_t piece_diff = black_pieces - white_pieces;
		std::uint32_t king_diff = black_kings - white_kings;
		std::uint32_t eval_diff = black_eval - white_eval;

		if (!white_pieces)
		{
			// black won
			return MaxUtility;
		}

		if (!black_pieces)
		{
			return MinUtility;
		}

		return piece_diff * 100 + king_diff * 10 + eval_diff;
	}

	BitBoard::utility_type BitBoard::GetWhiteUtility() const
	{
		std::uint32_t white_pieces = SWAR(white);
		std::uint32_t black_pieces = SWAR(black);
		std::uint32_t white_kings = SWAR(white & kings);
		std::uint32_t black_kings = SWAR(black & kings);
		std::uint32_t black_eval = EvaluatePosition(black);
		std::uint32_t white_eval = EvaluatePosition(white);

		std::uint32_t piece_diff = white_pieces - black_pieces;
		std::uint32_t king_diff = white_kings - black_kings;
		std::uint32_t eval_diff = white_eval - black_eval;

		if (!white_pieces)
		{
			// black won
			return MinUtility;
		}

		if (!black_pieces)
		{
			return MaxUtility;
		}

		return piece_diff * 100 + king_diff * 10 + eval_diff;
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
		const board_type white_kings = white & kings;
		board_type LL = (((not_occupied & R5Mask) >> 5) & white_kings) | ((not_occupied >> 4) & white_kings);
		board_type LR = (((not_occupied) >> 4) & white_kings) | (((not_occupied & R3Mask) >> 3) & white_kings);
		board_type UL = (((not_occupied & L5Mask) << 5) & white) | ((not_occupied << 4) & white);
		board_type UR = (((not_occupied << 4) & white) | ((not_occupied & L3Mask) << 3) & white);

		return  (LL | LR | UL | UR);
	}

	BitBoard::board_type BitBoard::GetBlackMoves() const
	{
		const board_type not_occupied = ~(black | white);
		const board_type black_kings = black & kings;
		board_type LL = (((not_occupied & R5Mask) >> 5) & black) | ((not_occupied >> 4) & black);
		board_type LR = (((not_occupied) >> 4) & black) | (((not_occupied & R3Mask) >> 3) & black);
		board_type UL = (((not_occupied & L5Mask) << 5) & black_kings) | ((not_occupied << 4) & black);
		board_type UR = (((not_occupied << 4) & black_kings) | ((not_occupied & L3Mask) << 3) & black_kings);

		return  (LL | LR | UL | UR);
	}

	BitBoard::board_type BitBoard::GetWhiteJumps() const
	{
		const board_type not_occupied = ~(black | white);
		const board_type white_kings = white & kings;
		board_type LL_even_to_odd = (((((not_occupied & R5Mask) >> 5) & black) >> 4) & white_kings);
		board_type LR_even_to_odd = (((((not_occupied >> 4) & black) & R3Mask) >> 3) & white_kings);
		board_type UL_even_to_odd = (((((not_occupied & L5Mask) << 5) & black) << 4) & white);
		board_type UR_even_to_odd = (((((not_occupied << 4) & black) & L3Mask) << 3) & white);

		board_type LL_odd_to_even = (((((not_occupied >> 4) & black) & R5Mask) >> 5) & white_kings);
		board_type LR_odd_to_even = (((((not_occupied & R3Mask) >> 3) & black) >> 4) & white_kings);
		board_type UL_odd_to_even = (((((not_occupied << 4) & black) & L5Mask) << 5) & white);
		board_type UR_odd_to_even = (((((not_occupied & L3Mask) << 3) & black) << 4) & white);

		board_type move = (LL_even_to_odd | LR_even_to_odd | UL_even_to_odd | UR_even_to_odd |
			LL_odd_to_even | LR_odd_to_even | UL_odd_to_even | UR_odd_to_even);
		printf("white jumps: %08x\n", move);
		return move;
	}

	BitBoard::board_type BitBoard::GetBlackJumps() const
	{
		const board_type not_occupied = ~(black | white);
		const board_type black_kings = black & kings;
		board_type LL_even_to_odd = (((((not_occupied & R5Mask) >> 5) & white) >> 4) & black);

		board_type LR_even_to_odd = (((((not_occupied >> 4) & white) & R3Mask) >> 3) & black);

		board_type UL_even_to_odd = (((((not_occupied & L5Mask) << 5) & white) << 4) & black_kings);
		
		board_type UR_even_to_odd = (((((not_occupied << 4) & white) & L3Mask) << 3) & black_kings);
		

		board_type LL_odd_to_even = (((((not_occupied >> 4) & white) & R5Mask) >> 5) & black);

		board_type LR_odd_to_even = (((((not_occupied & R3Mask) >> 3) & white) >> 4) & black);

		board_type UL_odd_to_even = (((((not_occupied << 4) & white) & L5Mask) << 5) & black_kings);

		board_type UR_odd_to_even = (((((not_occupied & L3Mask) << 3) & white) << 4) & black_kings);
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

#if 0
	std::uint32_t BitBoard::GetPossibleBlackMoves(Move *dst) const
	{
		board_type moves = GetBlackMoves();
		board_type jumps = GetBlackJumps();
		board_type empty = ~(black | white);

		int k = 0;
		board_type i = 1;

		if (!jumps)
		{
			while (moves && i && k < 48)
			{

				if (moves & i)
				{
					if (OddRows & i) // odd rows
					{
						if ((i << 4) & empty)//UL
						{
							dst[k++] = Move(BitBoard(white, (black & ~i) | (i << 4), (kings & ~i) | (((kings & i) << 4) | ((i << 4) & BlackKingMask))), false);

						}
						if (((i & L5Mask) << 5) & empty)//UR
						{

							dst[k++] = Move(BitBoard(white, (black & ~i) | (i << 5), (kings & ~i) | (((kings & i) << 5) | ((i << 5) & BlackKingMask))), false);

						}

						if (((i & kings) >> 4) & empty)//LL
						{
							dst[k++] = Move(BitBoard(white, (black & ~i) | (i >> 4), (kings & ~i) | (i >> 4)), false);

						}
						if ((((i&R3Mask) & kings) >> 3) & empty)//LR
						{
							dst[k++] = Move(BitBoard(white, (black & ~i) | (i >> 3), (kings & ~i) | (i >> 3)), false);

						}
					}
					else // even rows
					{
						if (((i & L3Mask) << 3) & empty)//UL
						{
							//

							dst[k++] = Move(BitBoard(white, (black & ~i) | (i << 3), (kings & ~i) | (((kings & i) << 3) | ((i << 3) & BlackKingMask))), false);

						}
						if ((i << 4) & empty)//UR
						{

							//
							dst[k++] = Move(BitBoard(white, (black & ~i) | (i << 4), (kings & ~i) | (((kings & i) << 4) | ((i << 4) & BlackKingMask))), false);

						}

						if ((((i  & kings)& R5Mask) >> 5) & empty)//LL
						{
							dst[k++] = Move(BitBoard(white, (black & ~i) | (i >> 5), (kings & ~i) | (i >> 5)), false);


						}
						if (((i  & kings) >> 4) & empty)//LR
						{
							dst[k++] = Move(BitBoard(white, (black & ~i) | (i >> 4), (kings & ~i) | (i >> 4)), false);


						}

					}
				}

				moves &= ~i;
				i = i << 1;
			}
		}
		else
		{

			while (jumps && i && k < 48)
			{
				if (jumps & i)
				{
					if (OddRows & i)
					{
						// odd rows, jump lands in odd row (enemy piece in even row)
						if ((i << 4) & white) // UL from odd
						{
							board_type j = i << 4;
							if (((j & L3Mask) << 3) & empty) // UL from even
							{
								dst[k++] = Move(BitBoard((white & ~j), (black & ~i) | (j << 3), (kings & (~i ^ j)) | ((((kings & i) << 4) << 3) | ((j << 3) & BlackKingMask))), true);
							}
						}

						if (((i & L5Mask) << 5) & white && k < 48) // UR from odd
						{
							board_type j = i << 5;
							if ((j << 4) & empty) // UR from even
							{
								dst[k++] = Move(BitBoard((white & ~j), (black & ~i) | (j << 4), (kings & (~i ^ j)) | ((((kings & i) << 5) << 4) | ((j << 4) & BlackKingMask))), true);
							}
						}

						if ((((i & R3Mask) & kings) >> 3) & white && k < 48) // LR from odd
						{
							board_type j = i >> 3;
							if ((j >> 4) & empty)// LR from even
							{
								dst[k++] = Move(BitBoard((white & ~j), (black & ~i) | (j >> 4), (kings & (~i ^ j)) | (j >> 4)), true);
							}
						}

						if (((i & kings) >> 4) & white && k < 48) // LL from odd
						{
							board_type j = i >> 4;
							if (((j & R5Mask) >> 5) & empty) // LL from even
							{
								dst[k++] = Move(BitBoard((white & ~j), (black & ~i) | (j >> 5), (kings & (~i ^ j)) | (j >> 5)), true);
							}
						}
					}
					else
					{
						// even rows
						if ((((i & L3Mask)) << 3) & white) // UL from even
						{
							board_type j = i << 3;
							if ((j << 4) & empty) // UL from odd
							{
								dst[k++] = Move(BitBoard((white & ~j), (black & ~i) | (j << 4), (kings & (~i ^ j)) | ((((kings & i) << 3) << 4) | ((j << 4) & BlackKingMask))), true);
							}
						}

						if ((i << 4) & white && k < 48) // UR from even
						{
							board_type j = i << 4;
							if (((j & L5Mask) << 5) & empty) // UR from odd
							{
								dst[k++] = Move(BitBoard((white & ~j), (black & ~i) | (j << 5), (kings & (~i ^ j)) | ((((kings & i) << 4) << 5) | ((j << 5) & BlackKingMask))), true);
							}
						}

						if (((i & kings) >> 4) & white && k < 48) // LR from even
						{
							board_type j = i >> 4;
							if (((j & R3Mask) >> 3) & empty)// LR from odd
							{
								dst[k++] = Move(BitBoard((white & ~j), (black & ~i) | (j >> 3), (kings & (~i ^ j)) | (j >> 3)), true);
							}
						}

						if ((((i & kings) & R5Mask) >> 5) & white && k < 48) // LL from even
						{
							board_type j = i >> 5;
							if ((j >> 4) & empty) // LL from odd
							{
								dst[k++] = Move(BitBoard((white & ~j), (black & ~i) | (j >> 4), (kings & (~i ^ j)) | (j >> 4)), true);
							}
						}
					}
				}

				jumps &= ~i;
				i = i << 1;
			}
		}
		return k;
	}

	std::uint32_t BitBoard::GetPossibleWhiteMoves(Move *dst) const
	{
		board_type moves = GetWhiteMoves();
		board_type jumps = GetWhiteJumps();
		board_type empty = ~(white | black);

		int k = 0;
		board_type i = 1;

		if (!jumps)
		{
			while (moves && i && k < 48)
			{

				if (moves & i)
				{

					if (OddRows & i) // odd rows
					{
						if (((i & kings) << 4) & empty)//UL
						{
							dst[k++] = Move(BitBoard((white & ~i) | (i << 4), black, (kings & ~i) | (i << 4)), false);

						}
						if (((((i & kings))& L5Mask) << 5) & empty)//UR
						{
							dst[k++] = Move(BitBoard((white & ~i) | (i << 5), black, (kings & ~i) | (i << 5)), false);

						}

						if ((i >> 4) & empty)//LL
						{
							dst[k++] = Move(BitBoard((white & ~i) | (i >> 4), black, (kings & ~i) | (((kings & i) >> 4) | ((i >> 4) & WhiteKingMask))), false);

						}
						if (((i & R3Mask) >> 3) & empty)//LR
						{
							dst[k++] = Move(BitBoard((white & ~i) | (i >> 3), black, (kings & ~i) | (((kings & i) >> 3) | ((i >> 3) & WhiteKingMask))), false);

						}

					}
					else // even rows
					{
						if ((((i & L3Mask) & kings) << 3) & empty) //UL
						{
							dst[k++] = Move(BitBoard((white & ~i) | (i << 3), black, (kings & ~i) | (i << 3)), false);

						}
						if (((i & kings) << 4) & empty) //UR
						{
							dst[k++] = Move(BitBoard((white & ~i) | (i << 4), black, (kings & ~i) | (i << 4)), false);

						}

						if (((i & R5Mask) >> 5) & empty) //LL
						{
							dst[k++] = Move(BitBoard((white & ~i) | (i >> 5), black, (kings & ~i) | (((kings & i) >> 5) | ((i >> 5) & WhiteKingMask))), false);
						}
						if ((i >> 4) & empty)//LR
						{
							dst[k++] = Move(BitBoard((white & ~i) | (i >> 4), black, (kings & ~i) | (((kings & i) >> 4) | ((i >> 4) & WhiteKingMask))), false);
						}
					}
				}

				moves &= ~i;
				i = i << 1;
			}
		}
		else
		{
			while (jumps && i && k < 48)
			{

				if (jumps & i)
				{
					if (OddRows & i) // odd rows
					{
						if (((i & kings) << 4) & black) //UL from odd
						{
							board_type j = i << 4;
							if (((j & L3Mask) << 3) & empty) // UL from even
							{
								dst[k++] = Move(BitBoard((white & ~i) | (j << 3), (black & ~j), (kings & (~i ^ j)) | (j << 3)), true);
							}
						}

						if ((((i & L5Mask)& kings) << 5) & black) //UR from odd
						{
							board_type j = i << 5;
							if ((j << 4) & empty) // UR from even
							{
								dst[k++] = Move(BitBoard((white & ~i) | (j << 4), (black & ~j), (kings & (~i ^ j)) | (j << 4)), true);
							}
						}

						if ((i >> 4) & black) //LL from odd
						{
							board_type j = i >> 4;
							if (((j & R5Mask) >> 5) & empty) // LL from even
							{
								dst[k++] = Move(BitBoard((white & ~i) | (j >> 5), (black & ~j), (kings & (~i ^ j)) | ((((kings & i) >> 4) >> 5) | ((j >> 5) & WhiteKingMask))), true);
							}
						}

						if (((i & R3Mask) >> 3) & black)//LR
						{
							board_type j = i >> 3;
							if ((j >> 4) & empty)
							{
								dst[k++] = Move(BitBoard((white & ~i) | (j >> 4), (black & ~j), (kings & (~i ^ j)) | ((((kings & i) >> 3) >> 4) | ((j >> 4) & WhiteKingMask))), true);
							}
						}
					}
					else // even rows
					{
						if ((((i & L3Mask) & kings) << 3) & black) //UL
						{
							board_type j = i << 3;
							if ((j << 4) & empty)
							{
								dst[k++] = Move(BitBoard((white & ~i) | (j << 4), (black & ~j), (kings & (~i ^ j)) | (j << 4)), true);
							}


						}
						if (((i & kings) << 4) & black) //UR
						{
							board_type j = i << 4;
							if (((j & L5Mask) << 5) & empty)
							{
								dst[k++] = Move(BitBoard((white & ~i) | (j << 5), (black & ~j), (kings & (~i ^ j)) | (j << 5)), true);
							}

						}

						if (((i & R5Mask) >> 5) & black) //LL
						{
							board_type j = i >> 5;
							if ((j >> 4) & empty)
							{
								dst[k++] = Move(BitBoard((white & ~i) | (j >> 4), (black & ~j), (kings & (~i ^ j)) | ((((kings & i) >> 5) >> 4) | ((j >> 4) & WhiteKingMask))), true);
							}
						}
						if ((i >> 4) & black)//LR
						{
							board_type j = i >> 4;
							if (((j & R3Mask) >> 3) & empty)
							{
								dst[k++] = Move(BitBoard((white & ~i) | (j >> 3), (black & ~j), (kings & (~i ^ j)) | ((((kings & i) >> 4) >> 3) | ((j >> 3) & WhiteKingMask))), true);
							}

						}
					}
				}

				jumps &= ~i;
				i = i << 1;
			}
		}
		return k;
	}
#endif
}