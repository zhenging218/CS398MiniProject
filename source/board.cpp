#include "precomp.h"
#include <utility>

namespace Checkers
{
	namespace
	{
		Piece *SetupBoard(unsigned int size)
		{
			unsigned int dim = size * size;
			Piece *ret = new Piece[dim];

			std::memset(ret, Piece::EMPTY, sizeof(Piece) * dim);

			unsigned int p1row = 0;
			unsigned int p2row = size - 1;
			unsigned int num_rows = (size - 2) / 2;

			unsigned int p1row_start_color = 0;
			unsigned int p2row_start_color = 1;

			for (unsigned int i = 0; i < num_rows; ++i)
			{
				for (unsigned int j = p1row_start_color; j < size; j += 2)
				{
					ret[i * size + j] = Piece::PLAYER1PAWN;
				}

				for (unsigned int j = p2row_start_color; j < size; j += 2)
				{
					ret[(p2row - i) * size + j] = Piece::PLAYER2PAWN;
				}

				p1row_start_color = 1 - p1row_start_color;
				p2row_start_color = 1 - p2row_start_color;
			}

			return ret;
		}

		Piece *DuplicateBoard(Piece const *src, unsigned int size)
		{
			unsigned int dim = size * size;
			Piece * ret = new Piece[dim];
			
			std::memcpy(ret, src, sizeof(Piece) * dim);

			return ret;
		}
	}

	Board::Board(unsigned int size) : board_size(size), board(SetupBoard(size))
	{
		ASSERT((size & 1) == 0, "Checkers Board Size must be a multiple of 2!");
	}

	Board::Board(Board const &src) : board_size(src.board_size), board(DuplicateBoard(src.board, src.board_size))
	{

	}

	Board::Board(Board &&src) : board_size(src.board_size), board(src.board)
	{
		src.board_size = 0;
		src.board = nullptr;
	}

	Board &Board::operator=(Board const &src)
	{
		return *this = Board(src);
	}

	Board &Board::operator=(Board &&src)
	{
		std::swap(board, src.board);
		std::swap(board_size, src.board_size);
		return *this;
	}

	Board::~Board()
	{
		delete board;
	}

	unsigned int Board::size() const
	{
		return board_size;
	}

	Piece const &Board::operator()(int row, int col) const
	{
		return board[row * board_size + col];
	}

	Piece &Board::operator()(int row, int col)
	{
		return board[row * board_size + col];
	}

	bool Board::Move(int row, int col, Movement const &move)
	{
#define PROCESS_MOVE(r, c, i, j)\
do \
{\
	unsigned int x = i, y = j;\
	if((r + y) < 0 || (r + y) >= board_size || (c + x) < 0 || (c + x) >= board_size)\
		return false;\
	unsigned int next = (r + y) * board_size + (c + x);\
	switch(board[next])\
	{\
		case Piece::PLAYER1PAWN:\
		if(curr & Piece::PLAYER1)\
			return false;\
		{\
			if((r + y + y) < 0 || (r + y + y) >= board_size || (c + x + x) < 0 || (c + x + x) >= board_size)\
				return false;\
			unsigned int next_next = (r + y + y) * board_size + (c + x + x);\
			if(board[next_next] == Piece::EMPTY)\
			{\
				std::swap(board[curr], board[next_next]);\
				board[next] = Piece::EMPTY;\
				if((r + y + y) == (board_size - 1))\
					board[next_next] = Piece::PLAYER1KING;\
				return true;\
			}\
		}\
		return false;\
		case Piece::PLAYER1KING:\
		if(curr & Piece::PLAYER1)\
			return false;\
		{\
			if((r + y + y) < 0 || (r + y + y) >= board_size || (c + x + x) < 0 || (c + x + x) >= board_size)\
				return false;\
			unsigned int next_next = (r + y + y) * board_size + (c + x + x);\
			if(board[next_next] == Piece::EMPTY)\
			{\
				std::swap(board[curr], board[next_next]);\
				board[next] = Piece::EMPTY;\
				return true;\
			}\
		}\
		return false;\
		case Piece::PLAYER2PAWN:\
		if(curr & Piece::PLAYER2)\
			return false;\
		{\
			if((r + y + y) < 0 || (r + y + y) >= board_size || (c + x + x) < 0 || (c + x + x) >= board_size)\
				return false;\
			unsigned int next_next = (r + y + y) * board_size + (c + x + x);\
			if(board[next_next] == Piece::EMPTY)\
			{\
				std::swap(board[curr], board[next_next]);\
				board[next] = Piece::EMPTY;\
				if((r + y + y) == 0)\
					board[next_next] = Piece::PLAYER2KING;\
				return true;\
			}\
		}\
		return false;\
		case Piece::PLAYER2KING:\
		if(curr & Piece::PLAYER2)\
			return false;\
		{\
			if((r + y + y) < 0 || (r + y + y) >= board_size || (c + x + x) < 0 || (c + x + x) >= board_size)\
				return false;\
			unsigned int next_next = (r + y + y) * board_size + (c + x + x);\
			if(board[next_next] == Piece::EMPTY)\
			{\
				std::swap(board[curr], board[next_next]);\
				board[next] = Piece::EMPTY;\
				return true;\
			}\
		}\
		return false;\
		case Piece::EMPTY:\
		std::swap(board[curr], board[next]);\
		if(board[next] & Piece::PAWN && (((board[next] & Piece::PLAYER2) && (r + y) == 0) || ((board[next] & Piece::PLAYER1) && (r + y) == (board_size - 1))))\
			board[next] = (Piece)(board[next] << 1);\
		return true;\
		default:\
		ASSERT(0, "Position on board at %d, %d has invalid value (%d)!", (int)r, (int)c, (int)board[next]);\
		break;\
	}\
} while(0)

		Piece curr = board[row * board_size + col];

		if (!(curr & Piece::KING) && !(((curr | move) & Movement::PLAYER1_PAWNMOVES) || ((curr | move) & Movement::PLAYER2_PAWNMOVES)))
		{
			return false;
		}

		switch (move)
		{
		case Movement::TOP_LEFT:
			PROCESS_MOVE(row, col, -1, 1);
			break;
		case Movement::TOP_RIGHT:
			PROCESS_MOVE(row, col, 1, 1);
			break;
		case Movement::BOTTOM_LEFT:
			PROCESS_MOVE(row, col, -1, -1);
			break;
		case Movement::BOTTOM_RIGHT:
			PROCESS_MOVE(row, col, 1, -1);
			break;
		default:
			ASSERT(0, "Checkers Move processing got a non-valid move (%d)!", (unsigned int)move);
			break;
		}

#undef PROCESS_MOVE

		return false;
	}

	std::ostream &operator<<(std::ostream &os, Board const &board)
	{
		
		for (unsigned int j = 0; j < board.board_size; ++j)
		{
			std::cout << "|";
			for (unsigned int i = 0; i < board.board_size; ++i)
			{
				Piece piece = board(j, i);
				if (piece & Piece::PLAYER1)
				{
					std::cout << (piece & Piece::KING ? "X" : " ") << "X|";
				}
				else if (piece & Piece::PLAYER2)
				{
					std::cout << (piece & Piece::KING ? "O" : " ") <<  "O|";
				}
				else
				{
					std::cout << "  |";
				}
			}
			std::cout << (j + 1 < board.board_size ? "\n" : "");
		}
		return os;
	}
}