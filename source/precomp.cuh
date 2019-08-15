#pragma once

#include <exception>
#include <string>
#include <helper_cuda.h>

#include "checkerrors.h"
#include "gpubitwise_helpers.cuh"
#include "gpubitboard.cuh"
#include "gpuminimax.cuh"
#include <stdio.h>

#define MAX(x,y) (((x) > (y)) ? (x) : (y))
#define MIN(x,y) (((x) < (y)) ? (x) : (y))