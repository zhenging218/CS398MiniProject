#include "precomp.cuh"

namespace Checkers
{
	__host__ __device__ std::uint32_t GPUSWAR32(std::uint32_t i)
	{
		i = i - ((i >> 1) & 0x55555555);
		i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
		return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
	}
}