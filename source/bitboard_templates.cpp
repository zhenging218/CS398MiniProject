#include "precomp.h"

#ifndef BITBOARD_TEMPLATES_CPP
#define BITBOARD_TEMPLATES_CPP

namespace Checkers
{
	template <typename OutIt>
	bool BitBoard::GetPossibleWhiteMoves(BitBoard const &b, OutIt &dst)
	{
		board_type moves = GetWhiteMoves(b);
		board_type jumps = GetWhiteJumps(b);
		board_type empty = ~(b.white | b.black);

		board_type i = 1;

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
							*(dst++) = BitBoard((b.white & ~i) | (i << 4), b.black, (b.kings & ~i) | (i << 4));

						}
						if (((((i & b.kings))& L5Mask) << 5) & empty)//UR
						{
							*(dst++) = BitBoard((b.white & ~i) | (i << 5), b.black, (b.kings & ~i) | (i << 5));

						}

						if ((i >> 4) & empty)//LL
						{
							*(dst++) = BitBoard((b.white & ~i) | (i >> 4), b.black, (b.kings & ~i) | (((b.kings & i) >> 4) | ((i >> 4) & WhiteKingMask)));

						}
						if (((i & R3Mask) >> 3) & empty)//LR
						{
							*(dst++) = BitBoard((b.white & ~i) | (i >> 3), b.black, (b.kings & ~i) | (((b.kings & i) >> 3) | ((i >> 3) & WhiteKingMask)));

						}

					}
					else // even rows
					{
						if ((((i & L3Mask) & b.kings) << 3) & empty) //UL
						{
							*(dst++) = BitBoard((b.white & ~i) | (i << 3), b.black, (b.kings & ~i) | (i << 3));

						}
						if (((i & b.kings) << 4) & empty) //UR
						{
							*(dst++) = BitBoard((b.white & ~i) | (i << 4), b.black, (b.kings & ~i) | (i << 4));

						}

						if (((i & R5Mask) >> 5) & empty) //LL
						{
							*(dst++) = BitBoard((b.white & ~i) | (i >> 5), b.black, (b.kings & ~i) | (((b.kings & i) >> 5) | ((i >> 5) & WhiteKingMask)));
						}
						if ((i >> 4) & empty)//LR
						{
							*(dst++) = BitBoard((b.white & ~i) | (i >> 4), b.black, (b.kings & ~i) | (((b.kings & i) >> 4) | ((i >> 4) & WhiteKingMask)));
						}
					}
				}

				moves &= ~i;
				i = i << 1;
			}
		}
		else
		{
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
								*(dst++) = BitBoard((b.white & ~i) | (j << 3), (b.black & ~j), (b.kings & (~i ^ j)) | (j << 3));
							}
						}

						if ((((i & L5Mask)& b.kings) << 5) & b.black) //UR from odd
						{
							board_type j = i << 5;
							if ((j << 4) & empty) // UR from even
							{
								*(dst++) = BitBoard((b.white & ~i) | (j << 4), (b.black & ~j), (b.kings & (~i ^ j)) | (j << 4));
							}
						}

						if ((i >> 4) & b.black) //LL from odd
						{
							board_type j = i >> 4;
							if (((j & R5Mask) >> 5) & empty) // LL from even
							{
								*(dst++) = BitBoard((b.white & ~i) | (j >> 5), (b.black & ~j), (b.kings & (~i ^ j)) | ((((b.kings & i) >> 4) >> 5) | ((j >> 5) & WhiteKingMask)));
							}
						}

						if (((i & R3Mask) >> 3) & b.black)//LR
						{
							board_type j = i >> 3;
							if ((j >> 4) & empty)
							{
								*(dst++) = BitBoard((b.white & ~i) | (j >> 4), (b.black & ~j), (b.kings & (~i ^ j)) | ((((b.kings & i) >> 3) >> 4) | ((j >> 4) & WhiteKingMask)));
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
								*(dst++) = BitBoard((b.white & ~i) | (j << 4), (b.black & ~j), (b.kings & (~i ^ j)) | (j << 4));
							}


						}
						if (((i & b.kings) << 4) & b.black) //UR
						{
							board_type j = i << 4;
							if (((j & L5Mask) << 5) & empty)
							{
								*(dst++) = BitBoard((b.white & ~i) | (j << 5), (b.black & ~j), (b.kings & (~i ^ j)) | (j << 5));
							}

						}

						if (((i & R5Mask) >> 5) & b.black) //LL
						{
							board_type j = i >> 5;
							if ((j >> 4) & empty)
							{
								*(dst++) = BitBoard((b.white & ~i) | (j >> 4), (b.black & ~j), (b.kings & (~i ^ j)) | ((((b.kings & i) >> 5) >> 4) | ((j >> 4) & WhiteKingMask)));
							}
						}
						if ((i >> 4) & b.black)//LR
						{
							board_type j = i >> 4;
							if (((j & R3Mask) >> 3) & empty)
							{
								*(dst++) = BitBoard((b.white & ~i) | (j >> 3), (b.black & ~j), (b.kings & (~i ^ j)) | ((((b.kings & i) >> 4) >> 3) | ((j >> 3) & WhiteKingMask)));
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

	template <typename OutIt>
	bool BitBoard::GetPossibleBlackMoves(BitBoard const &b, OutIt &dst)
	{
		board_type moves = GetBlackMoves(b);
		board_type jumps = GetBlackJumps(b);
		board_type empty = ~(b.black | b.white);

		board_type i = 1;

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
							*(dst++) = BitBoard(b.white, (b.black & ~i) | (i << 4), (b.kings & ~i) | (((b.kings & i) << 4) | ((i << 4) & BlackKingMask)));

						}
						if (((i & L5Mask) << 5) & empty)//UR
						{

							*(dst++) = BitBoard(b.white, (b.black & ~i) | (i << 5), (b.kings & ~i) | (((b.kings & i) << 5) | ((i << 5) & BlackKingMask)));

						}

						if (((i & b.kings) >> 4) & empty)//LL
						{
							*(dst++) = BitBoard(b.white, (b.black & ~i) | (i >> 4), (b.kings & ~i) | (i >> 4));

						}
						if ((((i&R3Mask) & b.kings) >> 3) & empty)//LR
						{
							*(dst++) = BitBoard(b.white, (b.black & ~i) | (i >> 3), (b.kings & ~i) | (i >> 3));

						}
					}
					else // even rows
					{
						if (((i & L3Mask) << 3) & empty)//UL
						{
							//

							*(dst++) = BitBoard(b.white, (b.black & ~i) | (i << 3), (b.kings & ~i) | (((b.kings & i) << 3) | ((i << 3) & BlackKingMask)));

						}
						if ((i << 4) & empty)//UR
						{

							//
							*(dst++) = BitBoard(b.white, (b.black & ~i) | (i << 4), (b.kings & ~i) | (((b.kings & i) << 4) | ((i << 4) & BlackKingMask)));

						}

						if ((((i  & b.kings)& R5Mask) >> 5) & empty)//LL
						{
							*(dst++) = BitBoard(b.white, (b.black & ~i) | (i >> 5), (b.kings & ~i) | (i >> 5));


						}
						if (((i  & b.kings) >> 4) & empty)//LR
						{
							*(dst++) = BitBoard(b.white, (b.black & ~i) | (i >> 4), (b.kings & ~i) | (i >> 4));


						}

					}
				}

				moves &= ~i;
				i = i << 1;
			}
		}
		else
		{
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
								*(dst++) = BitBoard((b.white & ~j), (b.black & ~i) | (j << 3), (b.kings & (~i ^ j)) | ((((b.kings & i) << 4) << 3) | ((j << 3) & BlackKingMask)));
							}
						}

						if (((i & L5Mask) << 5) & b.white) // UR from odd
						{
							board_type j = i << 5;
							if ((j << 4) & empty) // UR from even
							{
								*(dst++) = BitBoard((b.white & ~j), (b.black & ~i) | (j << 4), (b.kings & (~i ^ j)) | ((((b.kings & i) << 5) << 4) | ((j << 4) & BlackKingMask)));
							}
						}

						if ((((i & R3Mask) & b.kings) >> 3) & b.white) // LR from odd
						{
							board_type j = i >> 3;
							if ((j >> 4) & empty)// LR from even
							{
								*(dst++) = BitBoard((b.white & ~j), (b.black & ~i) | (j >> 4), (b.kings & (~i ^ j)) | (j >> 4));
							}
						}

						if (((i & b.kings) >> 4) & b.white) // LL from odd
						{
							board_type j = i >> 4;
							if (((j & R5Mask) >> 5) & empty) // LL from even
							{
								*(dst++) = BitBoard((b.white & ~j), (b.black & ~i) | (j >> 5), (b.kings & (~i ^ j)) | (j >> 5));
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
								*(dst++) = BitBoard((b.white & ~j), (b.black & ~i) | (j << 4), (b.kings & (~i ^ j)) | ((((b.kings & i) << 3) << 4) | ((j << 4) & BlackKingMask)));
							}
						}

						if ((i << 4) & b.white) // UR from even
						{
							board_type j = i << 4;
							if (((j & L5Mask) << 5) & empty) // UR from odd
							{
								*(dst++) = BitBoard((b.white & ~j), (b.black & ~i) | (j << 5), (b.kings & (~i ^ j)) | ((((b.kings & i) << 4) << 5) | ((j << 5) & BlackKingMask)));
							}
						}

						if (((i & b.kings) >> 4) & b.white) // LR from even
						{
							board_type j = i >> 4;
							if (((j & R3Mask) >> 3) & empty)// LR from odd
							{
								*(dst++) = BitBoard((b.white & ~j), (b.black & ~i) | (j >> 3), (b.kings & (~i ^ j)) | (j >> 3));
							}
						}

						if ((((i & b.kings) & R5Mask) >> 5) & b.white) // LL from even
						{
							board_type j = i >> 5;
							if ((j >> 4) & empty) // LL from odd
							{
								*(dst++) = BitBoard((b.white & ~j), (b.black & ~i) | (j >> 4), (b.kings & (~i ^ j)) | (j >> 4));
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
}

#endif