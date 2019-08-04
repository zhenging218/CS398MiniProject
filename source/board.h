#pragma once
#include <iostream>

namespace Checkers
{
	namespace Board
	{
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
	}
}