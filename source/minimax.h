#pragma once

#include "board.h"
#include "bitboard.h"

namespace Checkers
{
	class Minimax
	{
	private:
		static constexpr int default_search_depth = 8;
		static constexpr int default_max_turns = 50;

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

	public:

		using utility_type = std::int32_t;

		enum Turn : unsigned char
		{
			WHITE = Board::Piece::WHITE,
			BLACK = Board::Piece::BLACK
		};

		enum Result : char
		{
			INPROGRESS = -1,
			LOSE,
			DRAW
		};

		friend Minimax CreateMinimaxBoard(BitBoard const &src, Turn turn = Turn::WHITE);

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
		Turn GetTurn() const;

		

	private:

		BitBoard board;
		Turn turn;
		int turn_count;

		Minimax(BitBoard const &src, Turn t);

		static bool WhiteWinTest(BitBoard const &b);
		static bool WhiteLoseTest(BitBoard const &b);
		static bool BlackWinTest(BitBoard const &b);
		static bool BlackLoseTest(BitBoard const &b);

		// utility functions should be called for terminal test (i.e. before frontier generation).
		static bool GetBlackUtility(BitBoard const &b, utility_type &utility, int depth, int turns_left);
		static bool GetWhiteUtility(BitBoard const &b, utility_type &utility, int depth, int turns_left);

		static utility_type WhiteMoveMax(BitBoard const &b, int depth, int turns_left, utility_type alpha, utility_type beta);
		static utility_type WhiteMoveMin(BitBoard const &b, int depth, int turns_left, utility_type alpha, utility_type beta);
		static utility_type BlackMoveMax(BitBoard const &b, int depth, int turns_left, utility_type alpha, utility_type beta);
		static utility_type BlackMoveMin(BitBoard const &b, int depth, int turns_left, utility_type alpha, utility_type beta);
	};

	Minimax::Turn &operator++(Minimax::Turn &turn);
	Minimax::Turn operator++(Minimax::Turn &turn, int);

	Minimax::Turn &operator--(Minimax::Turn &turn) = delete;
	Minimax::Turn operator--(Minimax::Turn &turn, int) = delete;
}