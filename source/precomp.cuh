#ifndef GPU_PRECOMP_H
#define GPU_PRECOMP_H

#include <exception>
#include <string>
//#include <helper_cuda.h>

#include "checkerrors.h"
#include "gpubitwise_helpers.cuh"
#include "gpubitboard.cuh"
#include "gpuminimax.cuh"
#include <stdio.h>

#define GET_MAX(x,y) (((x) < (y)) ? (y) : (x))
#define GET_MIN(x,y) (((x) < (y)) ? (x) : (y))

#endif