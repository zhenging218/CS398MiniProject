#include "precomp.cuh"
#include "bitboard.h"
#include "gpuminimax.h"

#include <helper_cuda.h>
#include <cuda_runtime.h>

namespace Checkers
{
	namespace GPUMinimax
	{
		__global__ void master_white_next_kernel(int *placement, int X, GPUBitBoard const *boards, int num_boards, int depth, int turns)
		{
			int tx = threadIdx.x;
			__shared__ utility_type v[32];
			__shared__ utility_type *ret_v;
			__shared__ cudaStream_t streams[4];
			cudaEvent_t stream_start, stream_end;
			int t_placement;

			if (tx == 0)
			{
				t_placement = *placement;
				cudaMalloc(&ret_v, sizeof(utility_type) * num_boards);
				for (int i = 0; i < 4; ++i)
				{
					cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
				}
			}

			__syncthreads();

			if (tx < num_boards)
			{
				cudaEventCreate(&stream_start);
				cudaEventCreate(&stream_end);

				int e = tx % 4;
				cudaEventRecord(stream_start, streams[e]);
				master_white_min_kernel <<<dim3(1, 1, 1), dim3(32, 1, 1)>>> (ret_v + tx, boards, -Infinity, Infinity, depth, turns);
				cudaEventRecord(stream_end, streams[e]);

				cudaEventSynchronize(stream_end);
				cudaEventDestroy(stream_start);
				cudaEventDestroy(stream_end);

				v[tx] = ret_v[tx];
			}

			__syncthreads();

			if (tx == 0)
			{
				// all streams in the block should have completed processing by now.
				for (int i = 0; i < 4; ++i)
				{
					// cudaStreamSynchronize(streams[i]);
					cudaStreamDestroy(streams[i]);
				}

				for (int i = 0; i < num_boards; ++i)
				{
					if (X < v[i])
					{
						X = v[i];
						t_placement = i;
					}
				}

				*placement = t_placement;
			}

			__syncthreads();
		}

		__global__ void master_black_next_kernel(int *placement, int X, GPUBitBoard const *boards, int num_boards, int depth, int turns)
		{
			int tx = threadIdx.x;
			__shared__ utility_type v[32];
			__shared__ utility_type *ret_v;
			__shared__ cudaStream_t streams[4];
			cudaEvent_t stream_start, stream_end;
			int t_placement;

			if (tx == 0)
			{
				t_placement = *placement;
				cudaMalloc(&ret_v, sizeof(utility_type) * num_boards);
				for (int i = 0; i < 4; ++i)
				{
					cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
				}
			}

			__syncthreads();

			if (tx < num_boards)
			{
				cudaEventCreate(&stream_start);
				cudaEventCreate(&stream_end);

				int e = tx % 4;
				cudaEventRecord(stream_start, streams[e]);
				master_black_min_kernel << <dim3(1, 1, 1), dim3(32, 1, 1) >> > (ret_v + tx, boards, -Infinity, Infinity, depth, turns);
				cudaEventRecord(stream_end, streams[e]);

				cudaEventSynchronize(stream_end);
				cudaEventDestroy(stream_start);
				cudaEventDestroy(stream_end);

				v[tx] = ret_v[tx];
			}

			__syncthreads();

			if (tx == 0)
			{
				// all streams in the block should have completed processing by now.
				for (int i = 0; i < 4; ++i)
				{
					// cudaStreamSynchronize(streams[i]);
					cudaStreamDestroy(streams[i]);
				}

				for (int i = 0; i < num_boards; ++i)
				{
					if (X < v[i])
					{
						X = v[i];
						t_placement = i;
					}
				}

				*placement = t_placement;
			}

			__syncthreads();
		}
	}
}