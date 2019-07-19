#include "precomp.h"
#include <vector>

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
		max_turns();
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
		return false;
	}

	int Minimax::EvaluationTest() const
	{
		return 0;
	}

	int Minimax::Player1Move(int depth, int alpha, int beta) const
	{
		return 0;
	}

	int Minimax::Player2Move(int depth, int alpha, int beta) const
	{
		return 0;
	}

	bool Minimax::ProcessMove()
	{
		return false;
	}
}