#pragma once
#include "gpubitboard.cuh"
#include "minimax.h"
#include <thrust\device_vector.h>

namespace Checkers
{
	namespace GPUMinimax
	{
		enum GPUTurn : unsigned char
		{
			WHITE = Minimax::Turn::WHITE,
			BLACK = Minimax::Turn::BLACK
		};

		using utility_type = Minimax::utility_type;
		using board_type = GPUBitBoard::board_type;

		GPUTurn &operator--(GPUTurn &turn) = delete;
		GPUTurn operator--(GPUTurn &turn, int) = delete;

		__host__ __device__ GPUTurn &operator++(GPUTurn &turn);
		__host__ __device__ GPUTurn operator++(GPUTurn &turn, int);

		__device__ bool WhiteWinTest(GPUBitBoard const &b);
		__device__ bool WhiteLoseTest(GPUBitBoard const &b);
		__device__ bool BlackWinTest(GPUBitBoard const &b);
		__device__ bool BlackLoseTest(GPUBitBoard const &b);

		__device__ bool GetBlackUtility(GPUBitBoard const &b, utility_type &utility, int depth, int turns_left);
		__device__ bool GetWhiteUtility(GPUBitBoard const &b, utility_type &utility, int depth, int turns_left);

		__global__ utility_type WhiteMoveMax(GPUBitBoard const &b, int depth, int turns_left, utility_type alpha, utility_type beta);
		__global__ utility_type WhiteMoveMin(GPUBitBoard const &b, int depth, int turns_left, utility_type alpha, utility_type beta);
		__global__ utility_type BlackMoveMax(GPUBitBoard const &b, int depth, int turns_left, utility_type alpha, utility_type beta);
		__global__ utility_type BlackMoveMin(GPUBitBoard const &b, int depth, int turns_left, utility_type alpha, utility_type beta);

		__device__ bool GetMoves(GPUBitBoard const &board, GPUTurn turn, thrust::device_vector<GPUBitBoard> &ret);
		__device__ thrust::device_vector<GPUBitBoard> GenerateWhiteFrontier(GPUBitBoard const &board);
		__device__ thrust::device_vector<GPUBitBoard> GenerateBlackFrontier(GPUBitBoard const &board);
	}
}