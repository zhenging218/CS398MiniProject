#pragma once
#include <iostream>

namespace Checkers
{

	class Board
	{
	public:
		enum Piece : unsigned char
		{
			EMPTY = 0,
			PLAYER1PAWN = 0x1,
			PLAYER1KING = 0x2,
			PLAYER2PAWN = 0x4,
			PLAYER2KING = 0x8,
			PAWN = PLAYER1PAWN | PLAYER2PAWN,
			KING = PLAYER1KING | PLAYER2KING,
			PLAYER1 = PLAYER1PAWN | PLAYER1KING,
			PLAYER2 = PLAYER2PAWN | PLAYER2KING
		};

		enum Movement : unsigned char
		{
			NOMOVE = 0,
			TOP_LEFT = 0x10,
			TOP_RIGHT = 0x20,
			BOTTOM_LEFT = 0x40,
			BOTTOM_RIGHT = 0x80,
			PLAYER1_PAWNMOVES = TOP_LEFT | TOP_RIGHT | Piece::PLAYER1,
			PLAYER2_PAWNMOVES = BOTTOM_LEFT | BOTTOM_RIGHT | Piece::PLAYER2,
		};

		Board(unsigned int size = 8);
		Board(Board const &src);
		Board(Board &&src);
		Board &operator=(Board const &src);
		Board &operator=(Board &&src);
		~Board();

		unsigned int size() const;

		Piece const &operator()(int row, int col) const;
		Piece &operator()(int row, int col);

		bool Move(int row, int col, Movement const &move);
		bool FirstMove(int row, int col, Movement const &move);

		friend std::ostream &operator<<(std::ostream &os, Board const &board);
		
	private:
		unsigned int board_size;
		Piece *board;
	};
}