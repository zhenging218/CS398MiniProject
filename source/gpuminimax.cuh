#pragma once

#include "gpubitboard.cuh"

namespace Checkers
{
	namespace GPUMinimax
	{
		__global__ master_white_max_kernel(utility_type *v, utility_type *utility, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns);
		__global__ master_white_min_kernel(utility_type *v, utility_type *utility, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns);

		__global__ white_min_kernel(utility_type *v, GPUBitBoard const *src, int alpha, int beta, int depth, int turns);
		__global__ white_max_kernel(utility_type *v, GPUBitBoard const *src, int alpha, int beta, int depth, int turns);

		__global__ master_black_max_kernel(utility_type *v, utility_type *utility, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns);
		__global__ master_black_min_kernel(utility_type *v, utility_type *utility, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns);

		__global__ black_min_kernel(utility_type *v, GPUBitBoard const *src, int alpha, int beta, int depth, int turns);
		__global__ black_max_kernel(utility_type *v, GPUBitBoard const *src, int alpha, int beta, int depth, int turns);
	}
}