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
					ret[i * size + j] = Board::Piece::WHITEPAWN;
				}

				for (int j = p2row_start_color; j < size; j += 2)
				{
					ret[(p2row - i) * size + j] = Board::Piece::BLACKPAWN;
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

		int *DuplicatePositions(int const *src, int size)
		{
			int *ret = new int[size];
			std::memcpy(ret, src, sizeof(int) * size);
			return ret;
		}

		bool ProcessMove(int r, int c, int x, int y, Board::Piece *board, int size)
		{
			int currIndex = r * size + c;
			Board::Piece curr = board[currIndex];

			
		/*out of board*/
		if ((r + y) < 0 || (r + y) >= size || (c + x) < 0 || (c + x) >= size)
			return false; 
			int next = (r + y) * size + (c + x);
			switch (board[next])
			{
				/*check Next space*/
			case Board::Piece::WHITEPAWN:
			/*Current piece and next cannot be the same*/
			if (curr & Board::Piece::WHITE)
				return false; 
			{
				/*check next next piece if is within board if not return*/
				if ((r + y + y) < 0 || (r + y + y) >= size || (c + x + x) < 0 || (c + x + x) >= size)
					return false; 
					int next_next = (r + y + y) * size + (c + x + x); 
					/*if the next next space is empty, move over*/
					if (board[next_next] == Board::Piece::EMPTY)
					{
						std::swap(board[currIndex], board[next_next]);
						board[next] = Board::Piece::EMPTY; 
						if ((r + y + y) == (size - 1))
							board[next_next] = Board::Piece::WHITEKING; 
							return true; 
					}
			}
				return false; 
			case Board::Piece::WHITEKING:
			if (curr & Board::Piece::WHITE)
				return false; 
			{
				if ((r + y + y) < 0 || (r + y + y) >= size || (c + x + x) < 0 || (c + x + x) >= size)
					return false; 
					int next_next = (r + y + y) * size + (c + x + x); 
					if (board[next_next] == Board::Piece::EMPTY)
					{
						std::swap(board[currIndex], board[next_next]);
						board[next] = Board::Piece::EMPTY;
						return true; 
					}
			}
				return false; 
			case Board::Piece::BLACKPAWN:
			if (curr & Board::Piece::BLACK)
				return false; 
			{
				if ((r + y + y) < 0 || (r + y + y) >= size || (c + x + x) < 0 || (c + x + x) >= size)
					return false; 
					int next_next = (r + y + y) * size + (c + x + x); 
					if (board[next_next] == Board::Piece::EMPTY)
					{
						std::swap(board[currIndex], board[next_next]);
						board[next] = Board::Piece::EMPTY;
						if ((r + y + y) == 0)
							board[next_next] = Board::Piece::BLACKKING;
							return true; 
					}
			}
				return false; 
			case Board::Piece::BLACKKING:
			if (curr & Board::Piece::BLACK)
				return false; 
			{
				if ((r + y + y) < 0 || (r + y + y) >= size || (c + x + x) < 0 || (c + x + x) >= size)
					return false; 
					int next_next = (r + y + y) * size + (c + x + x); 
					if (board[next_next] == Board::Piece::EMPTY)
					{
						std::swap(board[currIndex], board[next_next]);
						board[next] = Board::Piece::EMPTY;
						return true; 
					}
			}
				return false; 
			case Board::Piece::EMPTY:
			std::swap(board[currIndex], board[next]);
			if (board[next] & Board::Piece::PAWN && (((board[next] & Board::Piece::BLACK) && (r + y) == 0) || ((board[next] & Board::Piece::WHITE) && (r + y) == (size - 1))))
				board[next] = (Board::Piece)(board[next] << 1);
				return true; 
			default:
			ASSERT(0, "Position on board at %d, %d has invalid value (%d)!", (int)r, (int)c, (int)board[next]); 
			break; 
			}

		}

		int *InitPiecePositionArray(Board::Piece const &src, int numPieces, Board::Piece const *board, int size)
		{
			int *ret = new int[numPieces];
			std::memset(ret, 0, sizeof(int) * sizeof(numPieces));
			int curr = 0;
			for (int j = 0; j < size && curr < numPieces; ++j)
			{
				for (int i = 0; i < size && curr < numPieces; ++i)
				{
					int position = j * size + i;
					if (board[position] == src)
					{
						ret[curr++] = position;
					}
				}
			}
			return ret;
		}
	}

	Board::Board(int numPieces, int size) : board_size(size), num_pieces(numPieces), board(SetupBoard(size)), player1Pieces(InitPiecePositionArray(Piece::WHITEPAWN, numPieces, board, size)), player2Pieces(InitPiecePositionArray(Piece::BLACKPAWN, numPieces, board, size))
	{
		ASSERT((size & 1) == 0, "Checkers Board Size must be a multiple of 2!");
	}

	Board::Board(int size) : Board((size * size - 2 * size) / 4, size)
	{
		
	}

	Board::Board(Board const &src) : board_size(src.board_size), num_pieces(src.num_pieces), board(DuplicateBoard(src.board, src.board_size)), player1Pieces(DuplicatePositions(src.player1Pieces, src.num_pieces)), player2Pieces(DuplicatePositions(src.player2Pieces, src.num_pieces))
	{

	}

	Board::Board(Board &&src) : board_size(src.board_size), num_pieces(src.num_pieces), board(src.board), player1Pieces(src.player1Pieces), player2Pieces(src.player2Pieces)
	{
		src.board_size = 0;
		src.num_pieces = 0;
		src.board = nullptr;
		src.player1Pieces = src.player2Pieces = nullptr;
	}

	Board &Board::operator=(Board const &src)
	{
		return *this = Board(src);
	}

	Board &Board::operator=(Board &&src)
	{
		std::swap(board, src.board);
		std::swap(board_size, src.board_size);
		std::swap(num_pieces, src.num_pieces);
		std::swap(player1Pieces, src.player1Pieces);
		std::swap(player2Pieces, src.player2Pieces);
		return *this;
	}

	Board::~Board()
	{
		delete board;
		delete player1Pieces;
		delete player2Pieces;
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
		Piece curr = board[row * board_size + col];
		if (!(curr & Piece::KING) && !(((curr | move) & Movement::WHITE_PAWNMOVES) || ((curr | move) & Movement::BLACK_PAWNMOVES)))
		{
			return false;
		}

		bool result = false;

		switch (move)
		{
		case Movement::TOP_LEFT:
			result = ProcessMove(row, col, -1, -1, board, board_size);
			break;
		case Movement::TOP_RIGHT:
			result = ProcessMove(row, col, 1, -1, board, board_size);
			break;
		case Movement::BOTTOM_LEFT:
			result = ProcessMove(row, col, -1, 1, board, board_size);
			break;
		case Movement::BOTTOM_RIGHT:
			result = ProcessMove(row, col, 1, 1, board, board_size);
			break;
		default:
			ASSERT(0, "Checkers Move processing got a non-valid move (%d)!", (int)move);
			break;
		}
		return result;
	}

	std::ostream &operator<<(std::ostream &os, Board const &board)
	{
		
		for (int i = 0; i < board.num_pieces; ++i)
		{
			std::cout << board.player1Pieces[i] << (i + 1 < board.num_pieces ? ", " : "\n");
		}

		for (int i = 0; i < board.num_pieces; ++i)
		{
			std::cout << board.player2Pieces[i] << (i + 1 < board.num_pieces ? ", " : "\n");
		}

		for (int j = 0; j < board.board_size; ++j)
		{
			std::cout << "|";
			for (int i = 0; i < board.board_size; ++i)
			{
				Board::Piece piece = board(j, i);
				if (piece & Board::Piece::WHITE)
				{
					std::cout << (piece & Board::Piece::KING ? "X" : " ") << "X|";
				}
				else if (piece & Board::Piece::BLACK)
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