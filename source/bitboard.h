#pragma once
#include <cstdint>
#include <vector>

namespace Checkers
{
	class BitBoard
	{
	public:
		using board_type = std::uint32_t;

	private:
		board_type white, black, kings;

		BitBoard(board_type w, board_type b, board_type k);

		board_type GetLLNonJumpWhiteMoves() const;
		board_type GetLRNonJumpWhiteMoves() const;
		board_type GetURNonJumpWhiteMmoves() const;
		board_type GetULNonJumpWhiteMoves() const;

		board_type GetLLJumpWhiteMoves() const;
		board_type GetLRJumpWhiteMoves() const;
		board_type GetURJumpWhiteMmoves() const;
		board_type GetULJumpWhiteMoves() const;


		board_type GetLLNonJumpBlackMoves() const;
		board_type GetLRNonJumpBlackMoves() const;
		board_type GetURNonJumpBlackMoves() const;
		board_type GetULNonJumpBlackMoves() const;

		board_type GetLLJumpBlackMoves() const;
		board_type GetLRJumpBlackMoves() const;
		board_type GetURJumpBlackMoves() const;
		board_type GetULJumpBlackMoves() const;
	public:

		BitBoard();

		board_type GetWhiteMoves() const;
		board_type GetBlackMoves() const;

		board_type GetWhiteJumps() const;
		board_type GetBlackJumps() const;

		board_type GetWhiteKings() const;
		board_type GetBlackKings() const;

		static std::vector<BitBoard> GetPossibleWhiteMoves(BitBoard const &src);
		static std::vector<BitBoard> GetPossibleBlackMoves(BitBoard const &src);
	};
}