#include "precomp.h"
#include <vector>
#include <algorithm>
#include <iterator>

namespace Checkers
{
	namespace
	{
		/*
		
		BitBoard::count_type EvaluatePosition(BitBoard::board_type board)
		{
			BitBoard::board_type corners = board & EvaluationMask;
			// BitBoard::board_type others = ~corners;
			// return SWAR(corners) * 2 + SWAR(others);
			return SWAR(corners);
		}

		BitBoard::count_type BitBoard::GetBlackUtility() const
		{
		std::uint32_t white_pieces = SWAR(white);
		std::uint32_t black_pieces = SWAR(black);
		std::uint32_t white_kings = SWAR(white & kings);
		std::uint32_t black_kings = SWAR(black & kings);
		std::uint32_t black_eval = EvaluatePosition(black);
		std::uint32_t white_eval = EvaluatePosition(white);

		std::uint32_t piece_diff = black_pieces - white_pieces;
		std::uint32_t king_diff = black_kings - white_kings;
		std::uint32_t eval_diff = black_eval - white_eval;

		if (!white_pieces)
		{
		// black won
		return MaxUtility;
		}

		if (!black_pieces)
		{
		return MinUtility;
		}

		return piece_diff * 100 + king_diff * 10 + eval_diff;
		}

		BitBoard::count_type BitBoard::GetWhiteUtility() const
		{
		std::uint32_t white_pieces = SWAR(white);
		std::uint32_t black_pieces = SWAR(black);
		std::uint32_t white_kings = SWAR(white & kings);
		std::uint32_t black_kings = SWAR(black & kings);
		std::uint32_t black_eval = EvaluatePosition(black);
		std::uint32_t white_eval = EvaluatePosition(white);

		std::uint32_t piece_diff = white_pieces - black_pieces;
		std::uint32_t king_diff = white_kings - black_kings;
		std::uint32_t eval_diff = white_eval - black_eval;

		if (!white_pieces)
		{
		// black won
		return MinUtility;
		}

		if (!black_pieces)
		{
		return MaxUtility;
		}

		return piece_diff * 100 + king_diff * 10 + eval_diff;
		}

		*/

		constexpr BitBoard::board_type EvaluationMask = 0x81188118u;

		std::pair<std::vector<Move>, bool> GetMoves(BitBoard const &board, Minimax::Turn turn)
		{
			std::vector<Move> ret;
			bool jumped = false;
			switch (turn)
			{
			case Minimax::Turn::PLAYER1:
			{
				auto result = board.GetPossibleWhiteMoves(std::back_insert_iterator<decltype(ret)>(ret));
				jumped = result.second;
			} break;
			case Minimax::Turn::PLAYER2:
			{
				auto result = board.GetPossibleBlackMoves(std::back_insert_iterator<decltype(ret)>(ret));
				jumped = result.second;
			} break;
			default:
				ASSERT(0, "GetMoves(%u) has wrong turn value!", turn);
				break;
			}
			return std::make_pair(ret, jumped);
		}
	}

	Minimax::utility_type Minimax::GetBlackUtility(BitBoard const &b)
	{
		utility_type black_pieces = b.GetBlackPieceCount();
		utility_type white_pieces = b.GetWhitePieceCount();
		utility_type black_kings = b.GetBlackKingsCount();
		utility_type white_kings = b.GetWhiteKingsCount();



		return 0;
	}

	Minimax::utility_type Minimax::GetWhiteUtility(BitBoard const &b)
	{

		return 0;
	}

	void Minimax::SetSearchDepth(int d)
	{
		depth() = d;
	}

	int Minimax::GetSearchDepth()
	{
		return depth();
	}

	void Minimax::SetMaxTurns(int t)
	{
		max_turns() = t;
	}

	int Minimax::GetMaxTurns()
	{
		return max_turns();
	}

	BitBoard const &Minimax::GetBoard() const
	{
		return board;
	}

	Minimax CreateMinimaxBoard(BitBoard const &src, Minimax::Turn turn)
	{
		return Minimax(src, turn);
	}

	Minimax::Result Minimax::Next()
	{
		if (!ProcessMove())
		{
			return Result::INVALID_MOVE;
		}

		if (WinTest())
		{
			return Result::WIN;
		}

		if (LoseTest())
		{
			return Result::LOSE;
		}

		if (DrawTest())
		{
			return Result::DRAW;
		}

		return Result::INPROGRESS;
	}

	Minimax::Result Minimax::Next(int row, int col, Board::Movement const &move)
	{
		return Result::DRAW;
	}

	Minimax::Minimax(BitBoard const &src, Turn t) : board(src), turn(t)
	{

	}

	// minimax functions

	bool Minimax::WinTest() const
	{
		return false;
	}

	bool Minimax::LoseTest() const
	{
		return false;
	}

	bool Minimax::DrawTest() const
	{
		return false;
	}

	bool Minimax::TerminalTest(int &terminal_value, int depth) const
	{
		if (WinTest())
			terminal_value = 1;
		else if (LoseTest())
			terminal_value = 0;
		else if (DrawTest())
			terminal_value = 0;
		else if (depth == 0)
			terminal_value = EvaluationTest();
		else
			return false;
		return true;
	}

	int Minimax::EvaluationTest() const
	{
		int winning_moves = 0;
		int left_diagonal_validity = 0;
		int right_diagonal_validity = 0;
		return 0;
	}

	int Minimax::Player1Move(int depth, int alpha, int beta) const
	{
		int v = -2;
		int terminalValue = 0;
		// check if need to stop the search
		if (TerminalTest(terminalValue, depth))
			return terminalValue;

		auto moves = GetMoves(board, Turn::PLAYER1);
		std::vector<Move> frontier;
		
		if (moves.second)
		{
			while (!moves.first.empty())
			{
				std::vector<Move> moves2;
				for (auto i = moves.first.begin(); i != moves.first.end(); ++i)
				{
					
					auto j = GetMoves(i->board, Turn::PLAYER1);
					if (j.second)
					{
						moves2.insert(moves2.end(), j.first.begin(), j.first.end());
					}
					else
					{
						frontier.push_back(*i);
					}
				}
				moves.first = std::move(moves2);
			}
		}
		else
		{
			frontier = std::move(moves.first);
		}

		for (auto const &move : frontier)
		{
			//Place(move.row, move.col, BoardType::Cross);
			v = std::max(Player2Move(depth - 1, alpha, beta), v);
			//UnPlace(move.row, move.col);
			if (v > beta)
			{
				// prune
				break;
			}
			alpha = std::max(alpha, v);
		}

		return v;
	}

	int Minimax::Player2Move(int depth, int alpha, int beta) const
	{
		int v = 2;
		int terminalValue = 0;
		// check if need to stop the search
		if (TerminalTest(terminalValue, depth))
			return terminalValue;

		auto moves = GetMoves(board, Turn::PLAYER2);
		std::vector<Move> frontier;

		if (moves.second)
		{
			while (!moves.first.empty())
			{
				std::vector<Move> moves2;
				for (auto i = moves.first.begin(); i != moves.first.end(); ++i)
				{

					auto j = GetMoves(i->board, Turn::PLAYER2);
					if (j.second)
					{
						moves2.insert(moves2.end(), j.first.begin(), j.first.end());
					}
					else
					{
						frontier.push_back(*i);
					}
				}
				moves.first = std::move(moves2);
			}
		}
		else
		{
			frontier = std::move(moves.first);
		}

		for (auto const &move : frontier)
		{
			//Place(move.row, move.col, BoardType::Cross);
			v = std::min(Player1Move(depth - 1, alpha, beta), v);
			//UnPlace(move.row, move.col);
			if (v > alpha)
			{
				// prune
				break;
			}
			beta = std::min(beta, v);
		}

		return v;
	}

	bool Minimax::ProcessMove()
	{
		return false;
	}
}