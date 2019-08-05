using gen_func = GPUBitBoard(*)(GPUBitBoard const *, GPUBitBoard::board_type);

using gen_jump_func = GPUBitBoard(*)(GPUBitBoard const *, GPUBitBoard::board_type, GPUBitBoard::board_type *);

__device__ GPUBitBoard GenWhiteMovesUL(GPUBitBoard const *b, GPUBitBoard::board_type i)
{
	if(position & 0xF0F0F0F0u) // odd
	{
		if(((i & b->kings) << 4) & empty)
		{
			return GPUBitBoard((b->white & ~i) | (i << 4), b->black, (b->kings & ~i) | (i << 4));
		}
	}
	else
	{
		if((((i & L3Mask) & b->kings) << 3) & empty)
		{
			return GPUBitBoard((b->white & ~i) | (i << 3), b->black, (b->kings & ~i) | (i << 3));
		}
	}
}

__device__ GPUBitBoard GenWhiteJumpsUL(GPUBitBoard const *b, GPUBitBoard::board_type i, GPUBitBoard::board_type *i2)
{
	if(position & 0xF0F0F0F0u) // odd
	{
		if (((i & b.kings) << 4) & b.black) //UL from odd
		{
			board_type j = i << 4;
			if (((j & L3Mask) << 3) & empty) // UL from even
			{
				*i2 = j << 3;
				return GPUBitBoard((b.white & ~i) | (j << 3), (b.black & ~j), (b.kings & (~i ^ j)) | (j << 3));
			}
		}
	}
	else
	{
		if ((((i & L3Mask) & b.kings) << 3) & b.black) //UL
		{
			board_type j = i << 3;
			if ((j << 4) & empty)
			{
				*i2 = j << 4;
				return GPUBitBoard((b.white & ~i) | (j << 4), (b.black & ~j), (b.kings & (~i ^ j)) | (j << 4));
			}


		}
	}
}

__device__ gen_func GenWhiteMoves = { GenWhiteMovesUL, GenWhiteMovesUR, GenWhiteMovesLL, GenWhiteMovesLR };
__device__ gen_func GenWhiteJumps = { GenWhiteJumpsUL, GenWhiteJumpsUR, GenWhiteJumpsLL, GenWhiteJumpsLR };

__device__ gen_func GenBlackMoves = { GenBlackMovesUL, GenBlackMovesUR, GenBlackMovesLL, GenBlackMovesLR };
__device__ gen_func GenBlackJumps = { GenBlackJumpsUL, GenBlackJumpsUR, GenBlackJumpsLL, GenBlackJumpsLR };

__global__ void EvaluateMoreWhiteJumpsKernel(GPUBitBoard::utility_type *utility, GPUBitBoard const *src, GPUBitBoard::board_type i, int depth, int turns_left)
{
	__shared__ GPUBitBoard *moves;
	
	int tx = threadIdx.x;
	
	GPUBitBoard board_type jumps = GPUBitBoard::GetWhiteJumps(src);
	
	if(jumps)
	{
		if(tx == 0)
		{
			moves = cudaMalloc(sizeof(GPUBitBoard) * 4);
		}
		
		__syncthreads();
		
		GPUBitBoard::board_type i2;
		moves[tx] = GetWhiteJumps[tx](src, /*  shift i according to tx (4 directions, based on odd/even row) */, &i2);
		EvaluateMoreWhiteJumps(/* utility return pointer */, moves[tx], i2, depth, turns_left);
	}
	else
	{
		// run next tree
		dim3 dimGrid(4, 1, 1);
		dim3 dimBlock(32, 1, 1);
		
		GPUBitBoard::utility_type *utility = cudaMalloc(sizeof(GPUBitBoard::utility_type));
		
		EvaluateWhiteMin<<<dimGrid, dimBlock>>>(utility, );
	}
}

__device__ void EvaluateMoreWhiteJumps(GPUBitBoard::utility_type *utility, GPUBitBoard const *src, GPUBitBoard::board_type i, int depth, int turns_left)
{
	dim3 dimGrid(1,1,1);
	dim3 dimBlock(4,1,1);
	EvaluateMoreWhiteJumpsKernel<<<dimGrid, dimBlock>>>(utility, src, i, depth, turns_left);
}

__global__ void StartMinimaxWhite(bool *valid, GPUBitBoard *dst, GPUBitBoard const *src, GPUBitBoard::board_type position, int depth, int turns_left)
{
	__shared__ GPUBitBoard *moves;
	
	int tx = threadIdx.x;
	int index = blockIdx.x * blockDim.x + tx;
	
	GPUBitBoard board_type moves = GPUBitBoard::GetWhiteMoves(src);
	GPUBitBoard board_type jumps = GPUBitBoard::GetWhiteJumps(src);
	
	if(jumps)
	{
		if(tx == 0)
		{
			moves = cudaMalloc(sizeof(GPUBitBoard) * 32);
		}
		__syncthreads();
		
		if(jumps & (1u << tx))
		{
			GPUBitBoard::board_type i2;
			moves[tx] = GenWhiteJumps[blockIdx.x](src, 1u << tx, &i2);
			// jump again (device function)
			EvaluateMoreWhiteJumps(/* utility return pointer */, moves[tx], i2, depth, turns_left);
			
		}
		
		__syncthreads();
		
		if(tx == 0)
		{
			cudaFree(moves);
			valid = true;
		}
		
		__syncthreads();
	}
	else if(moves)
	{
		if(tx == 0)
		{
			moves = cudaMalloc(sizeof(GPUBitBoard) * 32);
		}
		__syncthreads();
		
		if(moves & (1u << tx))
		{
			moves[tx] GenWhiteMoves[blockIdx.x](src, 1u << tx);
			// evaluate next player
		}
		
		__syncthreads();
		
		if(tx == 0)
		{
			cudaFree(moves);
			valid = true;
		}
		
		__syncthreads();
	}
	else
	{
		if(tx == 0)
		{
			*valid = false;
		}
	}
}