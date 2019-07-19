#include "precomp.h"
#include <utility>

namespace Checkers
{
	namespace
	{
		Board::Piece *SetupBoard(int size)
		{
			int dim = size * size;
			Board::Piece *ret = new Board::Piece[dim];

			std::memset(ret, Board::Piece::EMPTY, sizeof(Board::Piece) * dim);

			int p1row = 0;
			int p2row = size - 1;
			int num_rows = (size - 2) / 2;

			int p1row_start_color = 0;
			int p2row_start_color = 1;

			for (int i = 0; i < num_rows; ++i)
			{
				for (int j = p1row_start_color; j < size; j += 2)
				{
					ret[i * size + j] = Board::Piece::PLAYER1PAWN;
				}

				for (int j = p2row_start_color; j < size; j += 2)
				{
					ret[(p2row - i) * size + j] = Board::Piece::PLAYER2PAWN;
				}

				p1row_start_color = 1 - p1row_start_color;
				p2row_start_color = 1 - p2row_start_color;
			}

			return ret;
		}

		Board::Piece *DuplicateBoard(Board::Piece const *src, int size)
		{
			int dim = size * size;
			Board::Piece * ret = new Board::Piece[dim];
			
			std::memcpy(ret, src, sizeof(Board::Piece) * dim);

			return ret;
		}
	}

	Board::Board(int size) : board_size(size), board(SetupBoard(size))
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

	int Board::size() const
	{
		return board_size;
	}

	Board::Piece const &Board::operator()(int row, int col) const
	{
		return board[row * board_size + col];
	}

	Board::Piece &Board::operator()(int row, int col)
	{
		return board[row * board_size + col];
	}

	bool Board::Move(int row, int col, Movement const &move)
	{
#define PROCESS_MOVE(r, c, i, j)\
do \
{\
	int x = i, y = j;\
	/*out of board*/\
	if((r + y) < 0 || (r + y) >= board_size || (c + x) < 0 || (c + x) >= board_size)\
		return false;\
	int next = (r + y) * board_size + (c + x);\
	switch(board[next])\
	{\
		/*check Next space*/\
		case Piece::PLAYER1PAWN:\
		/*Current piece and next cannot be the same*/\
		if(curr & Piece::PLAYER1)\
			return false;\
		{\
			/*check next next piece if is within board if not return*/\
			if((r + y + y) < 0 || (r + y + y) >= board_size || (c + x + x) < 0 || (c + x + x) >= board_size)\
				return false;\
			int next_next = (r + y + y) * board_size + (c + x + x);\
			/*if the next next space is empty, move over*/\
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
			int next_next = (r + y + y) * board_size + (c + x + x);\
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
			int next_next = (r + y + y) * board_size + (c + x + x);\
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
			int next_next = (r + y + y) * board_size + (c + x + x);\
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
			ASSERT(0, "Checkers Move processing got a non-valid move (%d)!", (int)move);
			break;
		}

#undef PROCESS_MOVE

		return false;
	}

	std::ostream &operator<<(std::ostream &os, Board const &board)
	{
		
		for (int j = 0; j < board.board_size; ++j)
		{
			std::cout << "|";
			for (int i = 0; i < board.board_size; ++i)
			{
				Board::Piece piece = board(j, i);
				if (piece & Board::Piece::PLAYER1)
				{
					std::cout << (piece & Board::Piece::KING ? "X" : " ") << "X|";
				}
				else if (piece & Board::Piece::PLAYER2)
				{
					std::cout << (piece & Board::Piece::KING ? "O" : " ") <<  "O|";
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