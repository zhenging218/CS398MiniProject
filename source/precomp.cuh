#pragma once

#include <helper_cuda.h>

#include "gpubitwise_helpers.cuh"
#include "gpubitboard.cuh"
#include "gpuminimax.cuh"

#define CHECK_ERRORS()\
do\
{\
	cudaError err = cudaThreadSynchronize();\
	if (cudaSuccess != err)\
	{\
		std::cout << cudaGetErrorString(err) << "at " << __FILE__ << " (Line " << __LINE__ << ")" << std::endl;\
		throw;\
	}\
} while(0)