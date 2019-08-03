#include <cstdint>
#include <vector>

namespace Checkers
{
	class BitBoard
	{
		friend struct Move;

	public:
		using board_type = std::uint32_t;
		using count_type = std::int32_t;

	private:
		static constexpr board_type L3Mask = 0x0E0E0E0Eu;
		static constexpr board_type L5Mask = 0x00707070u;
		static constexpr board_type R3Mask = 0x70707070u;
		static constexpr board_type R5Mask = 0x0E0E0E00u;

		static constexpr board_type OddRows = 0xF0F0F0F0u;

		static constexpr board_type BlackKingMask = 0xF0000000u;
		static constexpr board_type WhiteKingMask = 0x0000000Fu;

		board_type white, black, kings;

		board_type GetBlackMoves() const;
		board_type GetWhiteMoves() const;

		board_type GetBlackJumps() const;
		board_type GetWhiteJumps() const;

	public:

		BitBoard();
		BitBoard(board_type w, board_type b, board_type k);

		friend std::ostream &operator<<(std::ostream &os, BitBoard const &src);

#if 0
		std::uint32_t GetPossibleWhiteMoves(Move *dst) const;
		std::uint32_t GetPossibleBlackMoves(Move *dst) const;
#endif

		template <typename OutIt>
		std::pair<OutIt, bool> GetPossibleWhiteMoves(OutIt dst) const;

		template <typename OutIt>
		std::pair<OutIt, bool> GetPossibleBlackMoves(OutIt dst) const;

		count_type GetBlackPieceCount() const;
		count_type GetWhitePieceCount() const;

		count_type GetBlackKingsCount() const;
		count_type GetWhiteKingsCount() const;
	};


	struct Move
	{
		BitBoard board;
		bool jump;

		Move();
		Move(BitBoard const &bb, bool j);
	};

#pragma region GetPossibleMoves

	template <typename OutIt>
	std::pair<OutIt, bool> BitBoard::GetPossibleWhiteMoves(OutIt dst) const
	{
		board_type moves = GetWhiteMoves();
		board_type jumps = GetWhiteJumps();
		board_type empty = ~(white | black);

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
						if (((i & kings) << 4) & empty)//UL
						{
							*(dst++) = Move(BitBoard((white & ~i) | (i << 4), black, (kings & ~i) | (i << 4)), false);

						}
						if (((((i & kings))& L5Mask) << 5) & empty)//UR
						{
							*(dst++) = Move(BitBoard((white & ~i) | (i << 5), black, (kings & ~i) | (i << 5)), false);

						}

						if ((i >> 4) & empty)//LL
						{
							*(dst++) = Move(BitBoard((white & ~i) | (i >> 4), black, (kings & ~i) | (((kings & i) >> 4) | ((i >> 4) & WhiteKingMask))), false);

						}
						if (((i & R3Mask) >> 3) & empty)//LR
						{
							*(dst++) = Move(BitBoard((white & ~i) | (i >> 3), black, (kings & ~i) | (((kings & i) >> 3) | ((i >> 3) & WhiteKingMask))), false);

						}

					}
					else // even rows
					{
						if ((((i & L3Mask) & kings) << 3) & empty) //UL
						{
							*(dst++) = Move(BitBoard((white & ~i) | (i << 3), black, (kings & ~i) | (i << 3)), false);

						}
						if (((i & kings) << 4) & empty) //UR
						{
							*(dst++) = Move(BitBoard((white & ~i) | (i << 4), black, (kings & ~i) | (i << 4)), false);

						}

						if (((i & R5Mask) >> 5) & empty) //LL
						{
							*(dst++) = Move(BitBoard((white & ~i) | (i >> 5), black, (kings & ~i) | (((kings & i) >> 5) | ((i >> 5) & WhiteKingMask))), false);
						}
						if ((i >> 4) & empty)//LR
						{
							*(dst++) = Move(BitBoard((white & ~i) | (i >> 4), black, (kings & ~i) | (((kings & i) >> 4) | ((i >> 4) & WhiteKingMask))), false);
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
						if (((i & kings) << 4) & black) //UL from odd
						{
							board_type j = i << 4;
							if (((j & L3Mask) << 3) & empty) // UL from even
							{
								*(dst++) = Move(BitBoard((white & ~i) | (j << 3), (black & ~j), (kings & (~i ^ j)) | (j << 3)), true);
							}
						}

						if ((((i & L5Mask)& kings) << 5) & black) //UR from odd
						{
							board_type j = i << 5;
							if ((j << 4) & empty) // UR from even
							{
								*(dst++) = Move(BitBoard((white & ~i) | (j << 4), (black & ~j), (kings & (~i ^ j)) | (j << 4)), true);
							}
						}

						if ((i >> 4) & black) //LL from odd
						{
							board_type j = i >> 4;
							if (((j & R5Mask) >> 5) & empty) // LL from even
							{
								*(dst++) = Move(BitBoard((white & ~i) | (j >> 5), (black & ~j), (kings & (~i ^ j)) | ((((kings & i) >> 4) >> 5) | ((j >> 5) & WhiteKingMask))), true);
							}
						}

						if (((i & R3Mask) >> 3) & black)//LR
						{
							board_type j = i >> 3;
							if ((j >> 4) & empty)
							{
								*(dst++) = Move(BitBoard((white & ~i) | (j >> 4), (black & ~j), (kings & (~i ^ j)) | ((((kings & i) >> 3) >> 4) | ((j >> 4) & WhiteKingMask))), true);
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
								*(dst++) = Move(BitBoard((white & ~i) | (j << 4), (black & ~j), (kings & (~i ^ j)) | (j << 4)), true);
							}


						}
						if (((i & kings) << 4) & black) //UR
						{
							board_type j = i << 4;
							if (((j & L5Mask) << 5) & empty)
							{
								*(dst++) = Move(BitBoard((white & ~i) | (j << 5), (black & ~j), (kings & (~i ^ j)) | (j << 5)), true);
							}

						}

						if (((i & R5Mask) >> 5) & black) //LL
						{
							board_type j = i >> 5;
							if ((j >> 4) & empty)
							{
								*(dst++) = Move(BitBoard((white & ~i) | (j >> 4), (black & ~j), (kings & (~i ^ j)) | ((((kings & i) >> 5) >> 4) | ((j >> 4) & WhiteKingMask))), true);
							}
						}
						if ((i >> 4) & black)//LR
						{
							board_type j = i >> 4;
							if (((j & R3Mask) >> 3) & empty)
							{
								*(dst++) = Move(BitBoard((white & ~i) | (j >> 3), (black & ~j), (kings & (~i ^ j)) | ((((kings & i) >> 4) >> 3) | ((j >> 3) & WhiteKingMask))), true);
							}

						}
					}
				}

				jumps &= ~i;
				i = i << 1;
			}
		}
		return std::make_pair(dst, jumped);
	}

	template <typename OutIt>
	std::pair<OutIt, bool> BitBoard::GetPossibleBlackMoves(OutIt dst) const
	{
		board_type moves = GetBlackMoves();
		board_type jumps = GetBlackJumps();
		board_type empty = ~(black | white);

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
							*(dst++) = Move(BitBoard(white, (black & ~i) | (i << 4), (kings & ~i) | (((kings & i) << 4) | ((i << 4) & BlackKingMask))), false);

						}
						if (((i & L5Mask) << 5) & empty)//UR
						{

							*(dst++) = Move(BitBoard(white, (black & ~i) | (i << 5), (kings & ~i) | (((kings & i) << 5) | ((i << 5) & BlackKingMask))), false);

						}

						if (((i & kings) >> 4) & empty)//LL
						{
							*(dst++) = Move(BitBoard(white, (black & ~i) | (i >> 4), (kings & ~i) | (i >> 4)), false);

						}
						if ((((i&R3Mask) & kings) >> 3) & empty)//LR
						{
							*(dst++) = Move(BitBoard(white, (black & ~i) | (i >> 3), (kings & ~i) | (i >> 3)), false);

						}
					}
					else // even rows
					{
						if (((i & L3Mask) << 3) & empty)//UL
						{
							//

							*(dst++) = Move(BitBoard(white, (black & ~i) | (i << 3), (kings & ~i) | (((kings & i) << 3) | ((i << 3) & BlackKingMask))), false);

						}
						if ((i << 4) & empty)//UR
						{

							//
							*(dst++) = Move(BitBoard(white, (black & ~i) | (i << 4), (kings & ~i) | (((kings & i) << 4) | ((i << 4) & BlackKingMask))), false);

						}

						if ((((i  & kings)& R5Mask) >> 5) & empty)//LL
						{
							*(dst++) = Move(BitBoard(white, (black & ~i) | (i >> 5), (kings & ~i) | (i >> 5)), false);


						}
						if (((i  & kings) >> 4) & empty)//LR
						{
							*(dst++) = Move(BitBoard(white, (black & ~i) | (i >> 4), (kings & ~i) | (i >> 4)), false);


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
						if ((i << 4) & white) // UL from odd
						{
							board_type j = i << 4;
							if (((j & L3Mask) << 3) & empty) // UL from even
							{
								*(dst++) = Move(BitBoard((white & ~j), (black & ~i) | (j << 3), (kings & (~i ^ j)) | ((((kings & i) << 4) << 3) | ((j << 3) & BlackKingMask))), true);
							}
						}

						if (((i & L5Mask) << 5) & white) // UR from odd
						{
							board_type j = i << 5;
							if ((j << 4) & empty) // UR from even
							{
								*(dst++) = Move(BitBoard((white & ~j), (black & ~i) | (j << 4), (kings & (~i ^ j)) | ((((kings & i) << 5) << 4) | ((j << 4) & BlackKingMask))), true);
							}
						}

						if ((((i & R3Mask) & kings) >> 3) & white) // LR from odd
						{
							board_type j = i >> 3;
							if ((j >> 4) & empty)// LR from even
							{
								*(dst++) = Move(BitBoard((white & ~j), (black & ~i) | (j >> 4), (kings & (~i ^ j)) | (j >> 4)), true);
							}
						}

						if (((i & kings) >> 4) & white) // LL from odd
						{
							board_type j = i >> 4;
							if (((j & R5Mask) >> 5) & empty) // LL from even
							{
								*(dst++) = Move(BitBoard((white & ~j), (black & ~i) | (j >> 5), (kings & (~i ^ j)) | (j >> 5)), true);
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
								*(dst++) = Move(BitBoard((white & ~j), (black & ~i) | (j << 4), (kings & (~i ^ j)) | ((((kings & i) << 3) << 4) | ((j << 4) & BlackKingMask))), true);
							}
						}

						if ((i << 4) & white) // UR from even
						{
							board_type j = i << 4;
							if (((j & L5Mask) << 5) & empty) // UR from odd
							{
								*(dst++) = Move(BitBoard((white & ~j), (black & ~i) | (j << 5), (kings & (~i ^ j)) | ((((kings & i) << 4) << 5) | ((j << 5) & BlackKingMask))), true);
							}
						}

						if (((i & kings) >> 4) & white) // LR from even
						{
							board_type j = i >> 4;
							if (((j & R3Mask) >> 3) & empty)// LR from odd
							{
								*(dst++) = Move(BitBoard((white & ~j), (black & ~i) | (j >> 3), (kings & (~i ^ j)) | (j >> 3)), true);
							}
						}

						if ((((i & kings) & R5Mask) >> 5) & white) // LL from even
						{
							board_type j = i >> 5;
							if ((j >> 4) & empty) // LL from odd
							{
								*(dst++) = Move(BitBoard((white & ~j), (black & ~i) | (j >> 4), (kings & (~i ^ j)) | (j >> 4)), true);
							}
						}
					}
				}

				jumps &= ~i;
				i = i << 1;
			}
		}
		return std::make_pair(dst, jumped);
	}

#pragma endregion
}