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
			WHITEPAWN = 0x1,
			WHITEKING = 0x2,
			BLACKPAWN = 0x4,
			BLACKKING = 0x8,
			PAWN = WHITEPAWN | BLACKPAWN,
			KING = WHITEKING | BLACKKING,
			WHITE = WHITEPAWN | WHITEKING,
			BLACK = BLACKPAWN | BLACKKING
		};

		enum Movement : unsigned char
		{
			NOMOVE = 0,
			TOP_LEFT = 0x10,
			TOP_RIGHT = 0x20,
			BOTTOM_LEFT = 0x40,
			BOTTOM_RIGHT = 0x80,
			WHITE_PAWNMOVES = TOP_LEFT | TOP_RIGHT | Piece::WHITE,
			BLACK_PAWNMOVES = BOTTOM_LEFT | BOTTOM_RIGHT | Piece::BLACK,
		};

		Board(int size = 8);
		Board(Board const &src);
		Board(Board &&src);
		Board &operator=(Board const &src);
		Board &operator=(Board &&src);
		~Board();

		int size() const;

		Piece const &operator()(int row, int col) const;
		Piece &operator()(int row, int col);

		bool Move(int row, int col, Movement const &move);

		friend std::ostream &operator<<(std::ostream &os, Board const &board);
		
	private:
		Board(int numPieces, int size);

		int board_size;
		int num_pieces;
		Piece *board;
		int *player1Pieces;
		int *player2Pieces;
	};
}