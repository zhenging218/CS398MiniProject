#pragma once

#include "gpubitboard.cuh"

namespace Checkers
{
	namespace GPUMinimax
	{
		__device__ extern GPUBitBoard::gen_move_func gen_white_move[2];
		__device__ extern GPUBitBoard::gen_move_func gen_black_move[2];

		using utility_type = Minimax::utility_type;

		static constexpr Minimax::utility_type PieceUtility = 1;
		static constexpr Minimax::utility_type KingsUtility = 3;

		static constexpr Minimax::utility_type MaxUtility = KingsUtility * 12;
		static constexpr Minimax::utility_type MinUtility = -MaxUtility;

		static constexpr Minimax::utility_type Infinity = 10000;

		static constexpr BitBoard::board_type EvaluationMask = 0x81188118u;

		__global__ void master_white_next_kernel(int *placement, utility_type X, GPUBitBoard const *boards, int num_boards, int depth, int turns);
		__global__ void master_black_next_kernel(int *placement, utility_type X, GPUBitBoard const *boards, int num_boards, int depth, int turns);

		__global__ void master_white_max_kernel(Minimax::utility_type *v, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns);
		__global__ void master_white_min_kernel(Minimax::utility_type *v, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns);

		__device__ utility_type white_min_device(GPUBitBoard src, int alpha, int beta, int depth, int turns);
		__device__ utility_type white_max_device(GPUBitBoard src, int alpha, int beta, int depth, int turns);

		__global__ void master_black_max_kernel(Minimax::utility_type *v, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns);
		__global__ void master_black_min_kernel(Minimax::utility_type *v, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns);

		__device__ utility_type black_min_device(GPUBitBoard src, int alpha, int beta, int depth, int turns);
		__device__ utility_type black_max_device(GPUBitBoard src, int alpha, int beta, int depth, int turns);

		__host__ __device__ bool GetWhiteUtility(GPUBitBoard const &src, Minimax::utility_type &terminal_value, int depth, int turns);
		__host__ __device__ bool GetBlackUtility(GPUBitBoard const &src, Minimax::utility_type &terminal_value, int depth, int turns);

		__host__ __device__ bool WhiteWinTest(BitBoard const &b);
		__host__ __device__ bool WhiteLoseTest(BitBoard const &b);
		__host__ __device__ bool BlackWinTest(BitBoard const &b);
		__host__ __device__ bool BlackLoseTest(BitBoard const &b);

		__host__ __device__ Minimax::utility_type WhiteMoveMax(BitBoard const &b, int depth, int turns_left, Minimax::utility_type alpha, Minimax::utility_type beta);
		__host__ __device__ Minimax::utility_type WhiteMoveMin(BitBoard const &b, int depth, int turns_left, Minimax::utility_type alpha, Minimax::utility_type beta);
		__host__ __device__ Minimax::utility_type BlackMoveMax(BitBoard const &b, int depth, int turns_left, Minimax::utility_type alpha, Minimax::utility_type beta);
		__host__ __device__ Minimax::utility_type BlackMoveMin(BitBoard const &b, int depth, int turns_left, Minimax::utility_type alpha, Minimax::utility_type beta);
	}
}