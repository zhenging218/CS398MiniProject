#include "precomp.h"
#include <vector>
#include <algorithm>
#include <iterator>

namespace Checkers
{
	namespace
	{
		using frontier_type = std::vector<BitBoard>;

		frontier_type GenerateWhiteFrontier(BitBoard const &board)
		{
			frontier_type frontier;

			BitBoard::GetPossibleWhiteMoves(board, std::back_insert_iterator<frontier_type>(frontier));

			return frontier;
		}

		frontier_type GenerateBlackFrontier(BitBoard const &board)
		{
			frontier_type frontier;
			BitBoard::GetPossibleBlackMoves(board, std::back_insert_iterator<frontier_type>(frontier));
			return frontier;
		}
	}

	Minimax::Turn &operator++(Minimax::Turn &turn)
	{
		turn = (turn == Minimax::WHITE ? Minimax::BLACK : Minimax::WHITE);
		return turn;
	}

	Minimax::Turn operator++(Minimax::Turn &turn, int)
	{
		Minimax::Turn ret = turn;
		++turn;
		return ret;
	}

	bool Minimax::BlackLoseTest(BitBoard const &b)
	{
		return !BitBoard::GetBlackPieceCount(b) || (!BitBoard::GetBlackMoves(b) && !BitBoard::GetBlackJumps(b));
	}

	bool Minimax::BlackWinTest(BitBoard const &b)
	{
		return !BitBoard::GetWhitePieceCount(b) || (!BitBoard::GetWhiteMoves(b) && !BitBoard::GetWhiteJumps(b));
	}

	bool Minimax::GetBlackUtility(BitBoard const &b, utility_type &utility, int depth, int turns_left)
	{
		utility_type black_pieces = BitBoard::GetBlackPieceCount(b) * PieceUtility;
		utility_type white_pieces = BitBoard::GetWhitePieceCount(b) * PieceUtility;

		utility_type black_kings = BitBoard::GetBlackKingsCount(b) * KingsUtility;
		utility_type white_kings = BitBoard::GetWhiteKingsCount(b) * KingsUtility;
		
		if(BlackWinTest(b))
		{
			utility = MaxUtility;
		}
		else if (BlackLoseTest(b))
		{
			utility = MinUtility;
		}
		else if (turns_left == 0)
		{
			if (black_pieces < white_pieces)
			{
				return MinUtility;
			}
			else if (white_pieces < black_pieces)
			{
				return MaxUtility;
			}
			else
			{
				utility = 0;
			}
		}
		else if (depth == 0)
		{
			utility = (black_pieces - white_pieces) + (black_kings - white_kings);
		}
		else
		{
			return false;
		}

		return true;
	}

	bool Minimax::WhiteWinTest(BitBoard const &b)
	{
		return !BitBoard::GetBlackPieceCount(b) || (!BitBoard::GetBlackMoves(b) && !BitBoard::GetBlackJumps(b));
	}

	bool Minimax::WhiteLoseTest(BitBoard const &b)
	{
		return !BitBoard::GetWhitePieceCount(b) || (!BitBoard::GetWhiteMoves(b) && !BitBoard::GetWhiteJumps(b));
	}

	bool Minimax::GetWhiteUtility(BitBoard const &b, utility_type &utility, int depth, int turns_left)
	{
		utility_type black_pieces = BitBoard::GetBlackPieceCount(b) * PieceUtility;
		utility_type white_pieces = BitBoard::GetWhitePieceCount(b) * PieceUtility;

		utility_type black_kings = BitBoard::GetBlackKingsCount(b) * KingsUtility;
		utility_type white_kings = BitBoard::GetWhiteKingsCount(b) * KingsUtility;

		if (WhiteWinTest(b))
		{
			utility = MaxUtility;
		}
		else if (WhiteLoseTest(b))
		{
			utility = MinUtility;
		}
		else if (turns_left == 0)
		{
			if (black_pieces < white_pieces)
			{
				return MaxUtility;
			}
			else if (white_pieces < black_pieces)
			{
				return MinUtility;
			}
			else
			{
				utility = 0;
			}
		}
		else if(depth == 0)
		{
			utility = (white_pieces - black_pieces) + (white_kings - black_kings);
		}
		else
		{
			return false;
		}
		return true;
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
		if (turn_count == 0)
		{
			return Result::DRAW;
		}

		if (turn == Turn::WHITE)
		{
			auto frontier = GenerateWhiteFrontier(board);
			if (frontier.empty())
			{
				return Result::LOSE;
			}
			int placement = -1;
			int size = (int)frontier.size();
			utility_type X = -Infinity;
			utility_type terminal_value = 0;

			for (int i = 0; i < size; ++i)
			{
				utility_type v = WhiteMoveMin(frontier[i], GetSearchDepth(), turn_count, -Infinity, Infinity);
				if (X < v)
				{
					X = v;
					placement = i;
				}
			}

			if (placement >= 0)
			{
				board = frontier[placement];
			}
		}
		else
		{
			auto frontier = GenerateBlackFrontier(board);
			if (frontier.empty())
			{
				return Result::LOSE;
			}
			int placement = -1;
			int size = (int)frontier.size();
			utility_type X = -Infinity;
			utility_type terminal_value = 0;

			for (int i = 0; i < size; ++i)
			{
				utility_type v = BlackMoveMin(frontier[i], GetSearchDepth(), turn_count, -Infinity, Infinity);
				if (X < v)
				{
					X = v;
					placement = i;
				}
			}

			if (placement >= 0)
			{
				board = frontier[placement];
			}
		}

		++turn;
		if (turn_count)
		{
			--turn_count;
		}
		return Result::INPROGRESS;
	}

	Minimax::Minimax(BitBoard const &src, Turn t) : board(src), turn(t), turn_count(max_turns())
	{

	}

	Minimax::utility_type Minimax::WhiteMoveMax(BitBoard const &b, int depth, int turns_left, utility_type alpha, utility_type beta)
	{
		utility_type v = -Infinity;
		utility_type terminal_value = 0;
		// check if need to stop the search
		if (GetWhiteUtility(b, terminal_value, depth, turns_left))
			return terminal_value;

		auto frontier = GenerateWhiteFrontier(b);

		for (auto const &move : frontier)
		{
			v = std::max(WhiteMoveMin(move, depth - 1, turns_left - 1, alpha, beta), v);
			if (v > beta)
			{
				// prune
				break;
			}
			alpha = std::max(alpha, v);
		}

		return v;
	}

	Minimax::utility_type Minimax::WhiteMoveMin(BitBoard const &b, int depth, int turns_left, utility_type alpha, utility_type beta) 
	{
		utility_type v = Infinity;
		utility_type terminal_value = 0;
		// check if need to stop the search
		if (GetWhiteUtility(b, terminal_value, depth, turns_left))
			return terminal_value;

		auto frontier = GenerateBlackFrontier(b);

		for (auto const &move : frontier)
		{
			v = std::min(WhiteMoveMax(move, depth - 1, turns_left - 1, alpha, beta), v);
			if (v < alpha)
			{
				// prune
				break;
			}
			beta = std::min(beta, v);
		}

		return v;
	}
																								  
	Minimax::utility_type Minimax::BlackMoveMax(BitBoard const &b, int depth, int turns_left, utility_type alpha, utility_type beta) 
	{
		utility_type v = -Infinity;
		utility_type terminal_value = 0;
		// check if need to stop the search
		if (GetBlackUtility(b, terminal_value, depth, turns_left))
			return terminal_value;

		auto frontier = GenerateBlackFrontier(b);

		for (auto const &move : frontier)
		{
			v = std::max(BlackMoveMin(move, depth - 1, turns_left - 1, alpha, beta), v);
			if (v > beta)
			{
				// prune
				break;
			}
			alpha = std::max(alpha, v);
		}

		return v;
	}
																								  
	Minimax::utility_type Minimax::BlackMoveMin(BitBoard const &b, int depth,int turns_left, utility_type alpha, utility_type beta) 
	{
		utility_type v = Infinity;
		utility_type terminal_value = 0;
		// check if need to stop the search
		if (GetBlackUtility(b, terminal_value, depth, turns_left))
			return terminal_value;
		auto frontier = GenerateWhiteFrontier(b);

		for (auto const &move : frontier)
		{
			v = std::min(BlackMoveMax(move, depth - 1, turns_left - 1, alpha, beta), v);
			if (v < alpha)
			{
				// prune
				break;
			}
			beta = std::min(beta, v);
		}

		return v;
	}

	Minimax::Turn Minimax::GetTurn() const
	{
		return turn;
	}
}