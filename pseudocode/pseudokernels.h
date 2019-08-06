__device__ GenWhiteMoves(GPUBitBoard::board_type cell, GPUBitBoard *out, GPUBitBoard const *board)
{
	// GPUBitBoard must have a validity boolean.
}

__device__ GenBlackMoves(GPUBitBoard::board_type cell, GPUBitBoard *out, GPUBitBoard const *board)
{
	// GPUBitBoard must have a validity boolean.
}

// will need master_black_max_kernel, master_black_min_kernel, black_max_kernel and black_min_kernel as well.
// non_master kernels will create 4 streams to use on its own per thread.

__global__ master_white_max_kernel(GPUBitBoard::utility_type *v, GPUBitBoard::utility_type *utility, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns)
{
	int tx = threadIdx.x;
	int index = blockIdx.x * blockDim.x + tx;
	
	// master kernel doesn't need terminal test because the CPU function that runs this kernel won't run kernel if terminal test terminates that node.
	
	if(index < num_boards)
	{
		// CPU will cudaMalloc the utility array for the master kernel.
		// CPU will initialize all the utility values to the first v on the leftmost tree that was evaluated by the CPU.
		white_min_kernel<<<dim3(1,1,1), dim3(32,1,1)>>>(utility + index, src + index, alpha, beta, depth - 1, turns - 1);
	}
	
	__syncthreads();
	
	if(index == 0)
	{
		// do ab-pruning
		for(int i = 0; i < num_boards; ++i)
		{
			*v = max(utility[index], *v);
			if(*v > beta)
			{
				break;
			}
		}
		alpha = max(alpha, *v);
	}
	
	__syncthreads();
}

__global__ master_white_min_kernel(GPUBitBoard::utility_type *v, GPUBitBoard::utility_type *utility, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns)
{
	// same logic as master_white_max_kernel, but use min node logic
}

__global__ white_min_kernel(GPUBitBoard::utility_type *v, GPUBitBoard const *src, int alpha, int beta, int depth, int turns)
{
	int tx = threadIdx.x;
	int t_beta = beta;
	__shared__ bool terminated;
	__shared__ GPUBitBoard::utility_type utilities[32];

	if(tx < 32)
	{
		utilities[tx] = *v;
	}
	
	__syncthreads();

	if(tx == 0)
	{
		GPUBitBoard::utility_type terminal_value = 0;
		if(src->valid)
		{
			*v = Infinity;
			terminated = GPUGetWhiteUtility(src, &terminal_value, depth, turns);
			if(terminated)
			{
				*v = terminal_value;
			}
		}
		else
		{
			terminated = true;
		}
	}
	
	__syncthreads();
	
	if(terminated)
	{
		return;
	}
	else
	{
		GPUBitBoard::utility_type *utility = cudaMalloc(sizeof(GPUBitBoard::utility_type) * 4);
		utility[0] = utility[1] = utility[2] = utility[3] = utilities[tx];
		GPUBitBoard *new_boards = cudaMalloc(sizeof(GPUBitBoard) * 4);
		
		cudaStream_t streams[4];
		cudaStreamCreateWithFlags(&streams[0], cudaStreamNonBlocking);
		cudaStreamCreateWithFlags(&streams[1], cudaStreamNonBlocking);
		cudaStreamCreateWithFlags(&streams[2], cudaStreamNonBlocking);
		cudaStreamCreateWithFlags(&streams[3], cudaStreamNonBlocking);
		
		GenBlackMoves(1u << tx, new_board, src);
		white_max_kernel<<<dim3(1,1,1), dim3(32,1,1), streams[0]>>>(utility, new_boards[0], t_alpha, t_beta, depth - 1, turns - 1);
		white_max_kernel<<<dim3(1,1,1), dim3(32,1,1), streams[1]>>>(utility + 1, new_boards[1], alpha, t_beta, depth - 1, turns - 1);
		white_max_kernel<<<dim3(1,1,1), dim3(32,1,1), streams[2]>>>(utility + 2, new_boards[2], alpha, t_beta, depth - 1, turns - 1);
		white_max_kernel<<<dim3(1,1,1), dim3(32,1,1), streams[3]>>>(utility + 3, new_boards[3], alpha, t_beta, depth - 1, turns - 1);
		
		// sync streams here
		
		for(int i  = 0; i < 4; ++i)
		{
			cudaStreamSynchronize(streams[i]);
			utilities[tx] = min(utility[i], utilities[tx]);
			if(utilities[tx] < alpha)
			{
				break;
			}
			else
			{
				t_beta = min(utilities[tx], t_beta);
			}
		}
		
		cudaFree(utility);
		cudaFree(new_boards);
		
		cudaStreamDestroy(streams[0]);
		cudaStreamDestroy(streams[1]);
		cudaStreamDestroy(streams[2]);
		cudaStreamDestroy(streams[3]);
		
		__syncthreads();
		
		if(tx == 0)
		{
			// final ab-pruning for this node
			for(int i = 0; i < 32; ++i)
			{
				*v = min(utilities[i], *v);
				if(*v < alpha)
				{
					break;
				}
				else
				{
					beta = min(utilities[i], beta);
				}
			}
		}
		
		__syncthreads();
	}
}

__global__ white_max_kernel(GPUBitBoard::utility_type *v, GPUBitBoard const *src, int alpha, int beta, int depth, int turns)
{
	// same as white_min_kernel, but use max logic.
}