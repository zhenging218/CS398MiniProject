#pragma once

#include <helper_cuda.h>
#include <device_functions.h>

namespace Checkers
{
	__device__ std::uint32_t GPUSWAR32(std::uint32_t i);
}
