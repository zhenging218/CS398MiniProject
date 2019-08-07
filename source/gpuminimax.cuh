#pragma once

#include "gpubitboard.cuh"

namespace Checkers
{
	namespace GPUMinimax
	{
		using utility_type = std::int32_t;

		static constexpr utility_type PieceUtility = 1;
		static constexpr utility_type KingsUtility = 3;

		static constexpr utility_type MaxUtility = KingsUtility * 12;
		static constexpr utility_type MinUtility = -MaxUtility;

		static constexpr utility_type Infinity = 10000;

		static constexpr BitBoard::board_type EvaluationMask = 0x81188118u;

		__global__ master_white_next_kernel(int *placement, int X, GPUBitBoard const *boards, int num_boards, int depth, int turns);
		__global__ master_black_next_kernel(int *placement, int X, GPUBitBoard const *boards, int num_boards, int depth, int turns);

		__global__ master_white_max_kernel(utility_type *v, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns);
		__global__ master_white_min_kernel(utility_type *v, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns);

		__global__ white_min_kernel(utility_type *v, GPUBitBoard src, int alpha, int beta, int depth, int turns);
		__global__ white_max_kernel(utility_type *v, GPUBitBoard src, int alpha, int beta, int depth, int turns);

		__global__ master_black_max_kernel(utility_type *v, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns);
		__global__ master_black_min_kernel(utility_type *v, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns);

		__global__ black_min_kernel(utility_type *v, GPUBitBoard src, int alpha, int beta, int depth, int turns);
		__global__ black_max_kernel(utility_type *v, GPUBitBoard src, int alpha, int beta, int depth, int turns);

		__host__ __device__ bool GetWhiteUtility(GPUBitBoard const &src, utility_type &terminal_value, int depth, int turns);
		__host__ __device__ bool GetBlackUtility(GPUBitBoard const &src, utility_type &terminal_value, int depth, int turns);

		__host__ __device__ bool WhiteWinTest(BitBoard const &b);
		__host__ __device__ bool WhiteLoseTest(BitBoard const &b);
		__host__ __device__ bool BlackWinTest(BitBoard const &b);
		__host__ __device__ bool BlackLoseTest(BitBoard const &b);

		__host__ utility_type WhiteMoveMax(BitBoard const &b, int depth, int turns_left, utility_type alpha, utility_type beta);
		__host__ utility_type WhiteMoveMin(BitBoard const &b, int depth, int turns_left, utility_type alpha, utility_type beta);
		__host__ utility_type BlackMoveMax(BitBoard const &b, int depth, int turns_left, utility_type alpha, utility_type beta);
		__host__ utility_type BlackMoveMin(BitBoard const &b, int depth, int turns_left, utility_type alpha, utility_type beta);
	}
}