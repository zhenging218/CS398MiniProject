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

	private:

		board_type white, black, kings;

		BitBoard(board_type w, board_type b, board_type k);

		board_type GetBlackMoves() const;
		board_type GetWhiteMoves() const;

		board_type GetBlackJumps() const;
		board_type GetWhiteJumps() const;

		BitBoard(board_type w, board_type b, board_type k);
	public:

		BitBoard();

		friend std::ostream &operator<<(std::ostream &os, BitBoard const &src);

		void GetPossibleWhiteMoves(Move *dst) const;
		void GetPossibleBlackMoves(Move *dst) const;
	};


	struct Move
	{
		BitBoard board;
		bool jump;

		Move(BitBoard const &bb, bool j);
	};
}