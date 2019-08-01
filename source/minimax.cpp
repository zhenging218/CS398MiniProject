#include "precomp.h"
#include <vector>
#include <algorithm>
namespace Checkers
{
	namespace
	{

		struct Move
		{
			int x, y;
			Board::Movement move;

			Move(int x_, int y_, Board::Movement const &m) : x(x_), y(y_), move(m)
			{

			}	
		};

		std::vector<Move> GetMoves(Board const &board)
		{
			std::vector<Move> ret;

			return ret;
		}
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

	Board const &Minimax::GetBoard() const
	{
		return board;
	}

	Minimax CreateMinimaxBoard(Board const &src, Minimax::Turn turn)
	{
		return Minimax(src, turn);
	}

	Minimax CreateMinimaxBoard(Board &&src, Minimax::Turn turn)
	{
		return Minimax(std::move(src), turn);
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

	Minimax::Minimax(Board const &src, Turn t) : board(src), turn(t)
	{

	}

	Minimax::Minimax(Board &&src, Turn t) : board(std::move(src)), turn(t)
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

		auto moves = GetMoves(board);
		for (auto const &move : moves)
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

		auto moves = GetMoves(board);
		for (auto const &move : moves)
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