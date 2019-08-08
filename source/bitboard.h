#pragma once

#include <cstdint>
#include <vector>

namespace Checkers
{
	struct BitBoard
	{
		using board_type = std::uint32_t;
		using count_type = std::int32_t;

		static constexpr board_type L3Mask = 0x0E0E0E0Eu;
		static constexpr board_type L5Mask = 0x00707070u;
		static constexpr board_type R3Mask = 0x70707070u;
		static constexpr board_type R5Mask = 0x0E0E0E00u;

		static constexpr board_type OddRows = 0xF0F0F0F0u;

		static constexpr board_type BlackKingMask = 0xF0000000u;
		static constexpr board_type WhiteKingMask = 0x0000000Fu;

		board_type white, black, kings;

		BitBoard();
		BitBoard(board_type w, board_type b, board_type k);

		friend std::ostream &operator<<(std::ostream &os, BitBoard const &src);

		static board_type GetBlackMoves(BitBoard const &b);
		static board_type GetWhiteMoves(BitBoard const &b);

		static board_type GetBlackJumps(BitBoard const &b);
		static board_type GetWhiteJumps(BitBoard const &b);

		template <typename OutIt>
		static void GetPossibleWhiteMoves(BitBoard const &b, OutIt &dst);

		template <typename OutIt>
		static void GetPossibleBlackMoves(BitBoard const &b, OutIt &dst);

		static count_type GetBlackPieceCount(BitBoard const &b);
		static count_type GetWhitePieceCount(BitBoard const &b);

		static count_type GetBlackKingsCount(BitBoard const &b);
		static count_type GetWhiteKingsCount(BitBoard const &b);

	private:
		template <typename OutIt>
		static void GetMoreWhiteJumps(BitBoard const &b, BitBoard::board_type i, OutIt &dst);

		template <typename OutIt>
		static void GetMoreBlackJumps(BitBoard const &b, BitBoard::board_type i, OutIt &dst);
	};
	
	bool operator==(BitBoard const &lhs, BitBoard const &rhs);
	bool operator!=(BitBoard const &lhs, BitBoard const &rhs);
}

#include "bitboard_templates.cpp"