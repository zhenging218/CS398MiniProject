#pragma once

#include "board.h"

namespace Checkers
{
	class Minimax
	{
	private:
		static constexpr int default_search_depth = 8;
		static constexpr int default_max_turns = 50;

	public:

		using utility_type = std::int32_t;

		enum Turn : unsigned char
		{
			PLAYER1 = Board::Piece::WHITE,
			PLAYER2 = Board::Piece::BLACK
		};

		enum Result : char
		{
			INVALID_MOVE = -2,
			INPROGRESS = -1,
			DRAW = 0,
			WIN,
			LOSE
		};

		friend Minimax CreateMinimaxBoard(BitBoard const &src, Turn turn);

		static void SetSearchDepth(int d);
		static int GetSearchDepth();

		static void SetMaxTurns(int t);
		static int GetMaxTurns();

		static constexpr int GetDefaultSearchDepth() noexcept
		{
			return default_search_depth;
		}

		static constexpr int GetDefaultMaxTurns() noexcept
		{
			return default_max_turns;
		}

		BitBoard const &GetBoard() const;

		Result Next();
		Result Next(int row, int col, Board::Movement const &move);

		// utility functions should be called by terminal test (i.e. before frontier generation).
		static utility_type GetBlackUtility(BitBoard const &b);
		static utility_type GetWhiteUtility(BitBoard const &b);

	private:
		static inline int &depth()
		{
			static int d = default_search_depth;
			return d;
		}

		static inline int &max_turns()
		{
			static int t = default_max_turns;
			return t;
		}

		BitBoard board;
		Turn turn;

		Minimax(BitBoard const &src, Turn t);

		bool WinTest() const;
		bool LoseTest() const;
		bool DrawTest() const;
		bool TerminalTest(int &terminal_value, int depth) const;
		int EvaluationTest() const;

		int Player1Move(int depth, int alpha, int beta) const;
		int Player2Move(int depth, int alpha, int beta) const;

		// minimax entry point
		bool ProcessMove();
	};

	
}