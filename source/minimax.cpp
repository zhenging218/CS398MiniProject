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

		bool GetMoves(BitBoard const &board, Minimax::Turn turn, std::vector<BitBoard> &ret)
		{
			ret.clear();
			bool jumped = false;
			switch (turn)
			{
			case Minimax::Turn::WHITE:
			{
				jumped = BitBoard::GetPossibleWhiteMoves(board, std::back_insert_iterator<std::vector<BitBoard>>(ret));
				
			} break;
			case Minimax::Turn::BLACK:
			{
				jumped = BitBoard::GetPossibleBlackMoves(board, std::back_insert_iterator<std::vector<BitBoard>>(ret));
			} break;
			default:
				ASSERT(0, "GetMoves(%u) has wrong turn value!", turn);
				break;
			}
			return jumped;
		}

		std::vector<BitBoard> GenerateWhiteFrontier(BitBoard const &board)
		{
			std::vector<BitBoard> frontier, curr_frontier;
			bool jumped = GetMoves(board, Minimax::Turn::WHITE, curr_frontier);

			if (jumped)
			{
				while (!curr_frontier.empty())
				{
					std::vector<BitBoard> new_frontier;
					for(BitBoard const &b : curr_frontier)
					{
						std::vector<BitBoard> temp_frontier;
						bool j = GetMoves(b, Minimax::Turn::WHITE, temp_frontier);
						if (j)
						{
							new_frontier.insert(new_frontier.end(), temp_frontier.begin(), temp_frontier.end());
						}
						else
						{
							frontier.push_back(b);
						}
					}
					curr_frontier = std::move(new_frontier);
				}
			}
			else
			{
				frontier = std::move(curr_frontier);
			}

			return frontier;
		}

		std::vector<BitBoard> GenerateBlackFrontier(BitBoard const &board)
		{
			std::vector<BitBoard> frontier, curr_frontier;
			bool jumped = GetMoves(board, Minimax::Turn::BLACK, curr_frontier);

			if (jumped)
			{
				while (!curr_frontier.empty())
				{
					std::vector<BitBoard> new_frontier;
					for (BitBoard const &b : curr_frontier)
					{
						std::vector<BitBoard> temp_frontier;
						bool j = GetMoves(b, Minimax::Turn::BLACK, temp_frontier);
						if (j)
						{
							new_frontier.insert(new_frontier.end(), temp_frontier.begin(), temp_frontier.end());
						}
						else
						{
							frontier.push_back(b);
						}
					}
					curr_frontier = std::move(new_frontier);
				}
			}
			else
			{
				frontier = std::move(curr_frontier);
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