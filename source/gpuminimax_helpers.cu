#include "precomp.cuh"
#include "bitboard.h"
#include "gpuminimax.h"

namespace Checkers
{
	namespace GPUMinimax
	{
		__host__ __device__ bool WhiteWinTest(GPUBitBoard const &b)
		{
			return !GPUBitBoard::GetBlackPieceCount(b) || (!GPUBitBoard::GetBlackMoves(b) && !GPUBitBoard::GetBlackJumps(b));
		}

		__host__ __device__ bool WhiteLoseTest(GPUBitBoard const &b)
		{
			return !GPUBitBoard::GetWhitePieceCount(b) || (!GPUBitBoard::GetWhiteMoves(b) && !GPUBitBoard::GetWhiteJumps(b));
		}

		__host__ __device__ bool BlackWinTest(GPUBitBoard const &b)
		{
			return !GPUBitBoard::GetWhitePieceCount(b) || (!GPUBitBoard::GetWhiteMoves(b) && !GPUBitBoard::GetWhiteJumps(b));
		}

		__host__ __device__ bool BlackLoseTest(GPUBitBoard const &b)
		{
			return !GPUBitBoard::GetBlackPieceCount(b) || (!GPUBitBoard::GetBlackMoves(b) && !GPUBitBoard::GetBlackJumps(b));
		}


		__host__ __device__ bool GetBlackUtility(GPUBitBoard const &b, Minimax::utility_type &utility, int depth, int turns_left)
		{
			Minimax::utility_type black_pieces = GPUBitBoard::GetBlackPieceCount(b) * PieceUtility;
			Minimax::utility_type white_pieces = GPUBitBoard::GetWhitePieceCount(b) * PieceUtility;

			Minimax::utility_type black_kings = GPUBitBoard::GetBlackKingsCount(b) * KingsUtility;
			Minimax::utility_type white_kings = GPUBitBoard::GetWhiteKingsCount(b) * KingsUtility;

			if (BlackWinTest(b))
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
				utility = (black_pieces - white_pieces) + (black_kings - white_kings);
			}
			else
			{
				return false;
			}
			return true;
		}

		__host__ __device__ bool GetWhiteUtility(GPUBitBoard const &b, Minimax::utility_type &utility, int depth, int turns_left)
		{
			Minimax::utility_type black_pieces = GPUBitBoard::GetBlackPieceCount(b) * PieceUtility;
			Minimax::utility_type white_pieces = GPUBitBoard::GetWhitePieceCount(b) * PieceUtility;

			Minimax::utility_type black_kings = GPUBitBoard::GetBlackKingsCount(b) * KingsUtility;
			Minimax::utility_type white_kings = GPUBitBoard::GetWhiteKingsCount(b) * KingsUtility;

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
			else if (depth == 0)
			{
				utility = (white_pieces - black_pieces) + (white_kings - black_kings);
			}
			else
			{
				return false;
			}
			return true;
		}
	}
}