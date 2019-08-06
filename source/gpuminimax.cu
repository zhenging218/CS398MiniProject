#include "precomp.cuh"
#include "gpuminimax.h"

namespace Checkers
{
	namespace GPUMinimax
	{
		// kernels
		__global__ void master_white_max_kernel(utility_type *v, utility_type *utility, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns)
		{

		}

		__global__ void master_white_min_kernel(utility_type *v, utility_type *utility, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns)
		{

		}

		__global__ void white_min_kernel(utility_type *v, GPUBitBoard const *src, int alpha, int beta, int depth, int turns)
		{

		}

		__global__ void white_max_kernel(utility_type *v, GPUBitBoard const *src, int alpha, int beta, int depth, int turns)
		{

		}

		__global__ void master_black_max_kernel(utility_type *v, utility_type *utility, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns)
		{

		}

		__global__ void master_black_min_kernel(utility_type *v, utility_type *utility, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns)
		{

		}

		__global__ void black_min_kernel(utility_type *v, GPUBitBoard const *src, int alpha, int beta, int depth, int turns)
		{

		}

		__global__ void black_max_kernel(utility_type *v, GPUBitBoard const *src, int alpha, int beta, int depth, int turns)
		{

		}


		// define the host functions here as well because it needs to run the kernels
		bool WhiteWinTest(BitBoard const &b)
		{

		}

		bool WhiteLoseTest(BitBoard const &b)
		{

		}

		bool BlackWinTest(BitBoard const &b)
		{

		}

		bool BlackLoseTest(BitBoard const &b)
		{

		}


		bool GetBlackUtility(BitBoard const &b, utility_type &utility, int depth, int turns_left)
		{

		}

		bool GetWhiteUtility(BitBoard const &b, utility_type &utility, int depth, int turns_left)
		{

		}

		utility_type WhiteMoveMax(BitBoard const &b, int depth, int turns_left, utility_type alpha, utility_type beta)
		{

		}

		utility_type WhiteMoveMin(BitBoard const &b, int depth, int turns_left, utility_type alpha, utility_type beta)
		{

		}

		utility_type BlackMoveMax(BitBoard const &b, int depth, int turns_left, utility_type alpha, utility_type beta)
		{

		}

		utility_type BlackMoveMin(BitBoard const &b, int depth, int turns_left, utility_type alpha, utility_type beta)
		{

		}
	}
}