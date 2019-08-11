#pragma once

#include <exception>
#include <string>
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
		char const *msg = cudaGetErrorString(err);\
		std::cout << msg << "at " << __FILE__ << " (Line " << __LINE__ << ")" << std::endl;\
		throw std::exception(msg);\
	}\
} while(0)