#include "precomp.h"
#include <vector>
#include <algorithm>
#include <iterator>

namespace Checkers
{
	namespace
	{
		
		constexpr Minimax::utility_type PieceUtility = 1;
		constexpr Minimax::utility_type KingsUtility = 3;
		constexpr Minimax::utility_type MaxUtility = KingsUtility * 12;
		constexpr Minimax::utility_type MinUtility = -MaxUtility;

		constexpr Minimax::utility_type Infinity = 10000;

		constexpr BitBoard::board_type EvaluationMask = 0x81188118u;

		std::pair<std::vector<Move>, bool> GetMoves(BitBoard const &board, Minimax::Turn turn)
		{
			std::vector<Move> ret;
			bool jumped = false;
			switch (turn)
			{
			case Minimax::Turn::WHITE:
			{
				auto result = board.GetPossibleWhiteMoves(std::back_insert_iterator<decltype(ret)>(ret));
				jumped = result.second;
			} break;
			case Minimax::Turn::BLACK:
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

		std::vector<Move> GenerateWhiteFrontier(BitBoard const &board)
		{
			auto moves = GetMoves(board, Minimax::Turn::WHITE);
			std::vector<Move> frontier;

			if (moves.second)
			{
				while (!moves.first.empty())
				{
					std::vector<Move> moves2;
					for (auto i = moves.first.begin(); i != moves.first.end(); ++i)
					{

						auto j = GetMoves(i->board, Minimax::Turn::WHITE);
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

			return frontier;
		}

		std::vector<Move> GenerateBlackFrontier(BitBoard const &board)
		{
			auto moves = GetMoves(board, Minimax::Turn::BLACK);
			std::vector<Move> frontier;

			if (moves.second)
			{
				while (!moves.first.empty())
				{
					std::vector<Move> moves2;
					for (auto i = moves.first.begin(); i != moves.first.end(); ++i)
					{

						auto j = GetMoves(i->board, Minimax::Turn::BLACK);
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

			return frontier;
		}
	}

	Minimax::Turn operator++(Minimax::Turn &turn)
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
		return !b.GetBlackPieceCount() || (!b.GetBlackMoves() && !b.GetBlackJumps());
	}

	bool Minimax::BlackWinTest(BitBoard const &b)
	{
		return !b.GetWhitePieceCount() || (!b.GetWhiteMoves() && !b.GetWhiteJumps());
	}

	bool Minimax::GetBlackUtility(BitBoard const &b, utility_type &utility, int depth, int turns_left)
	{
		utility_type black_pieces = b.GetBlackPieceCount() * PieceUtility;
		utility_type white_pieces = b.GetWhitePieceCount() * PieceUtility;

		
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
			utility = 0;
		}
		else if (depth == 0)
		{
			utility_type black_kings = b.GetBlackKingsCount() * KingsUtility;
			utility_type white_kings = b.GetWhiteKingsCount() * KingsUtility;

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
		return !b.GetBlackPieceCount() || (!b.GetBlackMoves() && !b.GetBlackJumps());
	}

	bool Minimax::WhiteLoseTest(BitBoard const &b)
	{
		return !b.GetWhitePieceCount() || (!b.GetWhiteMoves() && !b.GetWhiteJumps());
	}

	bool Minimax::GetWhiteUtility(BitBoard const &b, utility_type &utility, int depth, int turns_left)
	{
		utility_type black_pieces = b.GetBlackPieceCount() * PieceUtility;
		utility_type white_pieces = b.GetWhitePieceCount() * PieceUtility;

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
			utility = 0;
		}
		else if(depth == 0)
		{
			utility_type black_kings = b.GetBlackKingsCount() * KingsUtility;
			utility_type white_kings = b.GetWhiteKingsCount() * KingsUtility;
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

	Minimax CreateMinimaxBoard(BitBoard const &src, Minimax::Turn turn, int count)
	{
		return Minimax(src, turn, count);
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
				utility_type v = WhiteMoveMin(frontier[i].board, GetSearchDepth(), turn_count, -Infinity, Infinity);
				if (X < v)
				{
					X = v;
					placement = i;
				}
			}

			if (placement >= 0)
			{
				board = frontier[placement].board;
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
				utility_type v = BlackMoveMin(frontier[i].board, GetSearchDepth(), turn_count, -Infinity, Infinity);
				if (X < v)
				{
					X = v;
					placement = i;
				}
			}

			if (placement >= 0)
			{
				board = frontier[placement].board;
			}
		}

		++turn;
		if (turn_count)
		{
			--turn_count;
		}
		return Result::INPROGRESS;
	}

	Minimax::Minimax(BitBoard const &src, Turn t, int count) : board(src), turn(t), turn_count(count)
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
			v = std::max(WhiteMoveMin(move.board, depth - 1, turns_left - 1, alpha, beta), v);
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
			v = std::min(WhiteMoveMax(move.board, depth - 1, turns_left - 1, alpha, beta), v);
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
			v = std::max(BlackMoveMin(move.board, depth - 1, turns_left - 1, alpha, beta), v);
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
			v = std::min(BlackMoveMax(move.board, depth - 1, turns_left - 1, alpha, beta), v);
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