#pragma once

#include "gpubitboard.cuh"

namespace Checkers
{
	namespace GPUMinimax
	{
		enum NodeType
		{
			MAX,
			MIN
		};

		__host__ __device__ NodeType &operator++(NodeType &src);
		__host__ __device__ NodeType operator+(NodeType const &src, int i);

		NodeType &operator--(NodeType &src) = delete;
		NodeType operator-(NodeType const &src, int i) = delete;

		__device__ extern GPUBitBoard::gen_move_func gen_white_move[2];
		__device__ extern GPUBitBoard::gen_move_func gen_black_move[2];

		__device__ extern GPUBitBoard::gen_move_atomic_func gen_white_move_atomic[2];
		__device__ extern GPUBitBoard::gen_move_atomic_func gen_black_move_atomic[2];

		using utility_type = Minimax::utility_type;

		constexpr Minimax::utility_type PieceUtility = 1;
		constexpr Minimax::utility_type KingsUtility = 3;

		constexpr Minimax::utility_type MaxUtility = KingsUtility * 12;
		constexpr Minimax::utility_type MinUtility = -MaxUtility;

		constexpr Minimax::utility_type Infinity = 10000;

		constexpr BitBoard::board_type EvaluationMask = 0x81188118u;

		__global__ void white_next_kernel(int *placement, utility_type *v, utility_type X, GPUBitBoard const *boards, int num_boards, int depth, int turns);
		__global__ void black_next_kernel(int *placement, utility_type *v, utility_type X, GPUBitBoard const *boards, int num_boards, int depth, int turns);

		__device__ utility_type explore_black_frontier(GPUBitBoard const &board, utility_type alpha, utility_type beta, NodeType node_type, int depth, int turns);
		__device__ utility_type explore_white_frontier(GPUBitBoard const &board, utility_type alpha, utility_type beta, NodeType node_type, int depth, int turns);

		__global__ void black_kernel(utility_type *v, utility_type X, GPUBitBoard const *boards, int num_boards, utility_type alpha, utility_type beta, NodeType node_type, int depth, int turns);
		__global__ void white_kernel(utility_type *v, utility_type X, GPUBitBoard const *boards, int num_boards, utility_type alpha, utility_type beta, NodeType node_type, int depth, int turns);

		__host__ __device__ bool GetWhiteUtility(GPUBitBoard const &src, Minimax::utility_type &terminal_value, int depth, int turns);
		__host__ __device__ bool GetBlackUtility(GPUBitBoard const &src, Minimax::utility_type &terminal_value, int depth, int turns);

		__host__ __device__ bool WhiteWinTest(BitBoard const &b);
		__host__ __device__ bool WhiteLoseTest(BitBoard const &b);
		__host__ __device__ bool BlackWinTest(BitBoard const &b);
		__host__ __device__ bool BlackLoseTest(BitBoard const &b);

		__host__ Minimax::utility_type WhiteMoveMax(BitBoard const &b, int depth, int turns_left, Minimax::utility_type alpha, Minimax::utility_type beta);
		__host__ Minimax::utility_type WhiteMoveMin(BitBoard const &b, int depth, int turns_left, Minimax::utility_type alpha, Minimax::utility_type beta);
		__host__ Minimax::utility_type BlackMoveMax(BitBoard const &b, int depth, int turns_left, Minimax::utility_type alpha, Minimax::utility_type beta);
		__host__ Minimax::utility_type BlackMoveMin(BitBoard const &b, int depth, int turns_left, Minimax::utility_type alpha, Minimax::utility_type beta);
	}
}