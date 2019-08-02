#pragma once
#include <cstdint>
#include <vector>

namespace Checkers
{
	class BitBoard
	{
		friend struct Move;

	public:
		using board_type = std::uint32_t;
		using utility_type = std::int32_t;

	private:

		board_type white, black, kings;

		board_type GetBlackMoves() const;
		board_type GetWhiteMoves() const;

		board_type GetBlackJumps() const;
		board_type GetWhiteJumps() const;

	public:

		BitBoard();
		BitBoard(board_type w, board_type b, board_type k);

		friend std::ostream &operator<<(std::ostream &os, BitBoard const &src);

		std::uint32_t GetPossibleWhiteMoves(Move *dst) const;
		std::uint32_t GetPossibleBlackMoves(Move *dst) const;

		utility_type GetBlackUtility() const;
		utility_type GetWhiteUtility() const;

		utility_type GetBlackPieceCount() const;
		utility_type GetWhitePieceCount() const;

	};


	struct Move
	{
		BitBoard board;
		bool jump;

		Move();
		Move(BitBoard const &bb, bool j);
	};
}