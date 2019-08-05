#include "precomp.cuh"

namespace Checkers
{
	namespace GPUMinimax
	{
		constexpr Minimax::utility_type PieceUtility = 1;
		constexpr Minimax::utility_type KingsUtility = 3;
		constexpr Minimax::utility_type MaxUtility = KingsUtility * 12;
		constexpr Minimax::utility_type MinUtility = -MaxUtility;

		constexpr Minimax::utility_type Infinity = 10000;

		constexpr BitBoard::board_type EvaluationMask = 0x81188118u;

		__host__ __device__ GPUTurn &operator++(GPUTurn &turn)
		{

		}

		__host__ __device__ GPUTurn operator++(GPUTurn &turn, int)
		{

		}

		__device__ bool WhiteWinTest(GPUBitBoard const &b)
		{

		}

		__device__ bool WhiteLoseTest(GPUBitBoard const &b)
		{

		}

		__device__ bool BlackWinTest(GPUBitBoard const &b)
		{

		}

		__device__ bool BlackLoseTest(GPUBitBoard const &b)
		{

		}

		__device__ bool GetBlackUtility(GPUBitBoard const &b, utility_type &utility, int depth, int turns_left)
		{

		}

		__device__ bool GetWhiteUtility(GPUBitBoard const &b, utility_type &utility, int depth, int turns_left)
		{

		}

		__device__ utility_type WhiteMoveMax(GPUBitBoard const &b, int depth, int turns_left, utility_type alpha, utility_type beta)
		{

		}

		__device__ utility_type WhiteMoveMin(GPUBitBoard const &b, int depth, int turns_left, utility_type alpha, utility_type beta)
		{

		}

		__device__ utility_type BlackMoveMax(GPUBitBoard const &b, int depth, int turns_left, utility_type alpha, utility_type beta)
		{

		}

		__device__ utility_type BlackMoveMin(GPUBitBoard const &b, int depth, int turns_left, utility_type alpha, utility_type beta)
		{

		}

		__device__ bool GetMoves(GPUBitBoard const &board, GPUTurn turn, thrust::device_vector<GPUBitBoard> &ret)
		{

		}

		__device__ thrust::device_vector<GPUBitBoard> GenerateWhiteFrontier(GPUBitBoard const &board)
		{

		}

		__device__ thrust::device_vector<GPUBitBoard> GenerateBlackFrontier(GPUBitBoard const &board)
		{

		}
	}
}