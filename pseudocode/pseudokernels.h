__device__ GPUBitBoard::board_type board GetWhiteJumps(GPUBitBoard const *b)
{
	const board_type not_occupied = ~(b->black | b->white);
	const board_type white_kings = b->white & b->kings;
	board_type LL_even_to_odd = (((((not_occupied & R5Mask) >> 5) & b->black) >> 4) & white_kings);
	board_type LR_even_to_odd = (((((not_occupied >> 4) & b->black) & R3Mask) >> 3) & white_kings);
	board_type UL_even_to_odd = (((((not_occupied & L5Mask) << 5) & b->black) << 4) & b->white);
	board_type UR_even_to_odd = (((((not_occupied << 4) & b->black) & L3Mask) << 3) & b->white);

	board_type LL_odd_to_even = (((((not_occupied >> 4) & b->black) & R5Mask) >> 5) & white_kings);
	board_type LR_odd_to_even = (((((not_occupied & R3Mask) >> 3) & b->black) >> 4) & white_kings);
	board_type UL_odd_to_even = (((((not_occupied << 4) & b->black) & L5Mask) << 5) & b->white);
	board_type UR_odd_to_even = (((((not_occupied & L3Mask) << 3) & b->black) << 4) & b->white);

	board_type move = (LL_even_to_odd | LR_even_to_odd | UL_even_to_odd | UR_even_to_odd |
		LL_odd_to_even | LR_odd_to_even | UL_odd_to_even | UR_odd_to_even);
	return move;
}

__device__ GPUBitBoard::board_type board GetBlackJumps(GPUBitBoard const *b)
{
	const board_type not_occupied = ~(b->black | b->white);
	const board_type black_kings = b->black & b->kings;
	board_type LL_even_to_odd = (((((not_occupied & R5Mask) >> 5) & b->white) >> 4) & b->black);
	board_type LR_even_to_odd = (((((not_occupied >> 4) & b->white) & R3Mask) >> 3) & b->black);
	board_type UL_even_to_odd = (((((not_occupied & L5Mask) << 5) & b->white) << 4) & black_kings);
	board_type UR_even_to_odd = (((((not_occupied << 4) & b->white) & L3Mask) << 3) & black_kings);
	
	board_type LL_odd_to_even = (((((not_occupied >> 4) & b->white) & R5Mask) >> 5) & b->black);
	board_type LR_odd_to_even = (((((not_occupied & R3Mask) >> 3) & b->white) >> 4) & b->black);
	board_type UL_odd_to_even = (((((not_occupied << 4) & b->white) & L5Mask) << 5) & black_kings);
	board_type UR_odd_to_even = (((((not_occupied & L3Mask) << 3) & b->white) << 4) & black_kings);
	
	board_type move = (LL_even_to_odd | LR_even_to_odd | UL_even_to_odd | UR_even_to_odd |
		LL_odd_to_even | LR_odd_to_even | UL_odd_to_even | UR_odd_to_even);
	return move;
}

__device__ void GenWhiteMove(GPUBitBoard::board_type cell, GPUBitBoard *out, GPUBitBoard const *board)
{
	// GPUBitBoard must have a validity boolean.
	//normal move (UL, UR)
	//king extra moves (LL, LR)
	GPUBitBoard::board_type empty = ~(board->white | board->black);

	if(GPUBitBoard::oddrow & cell)
	{
		//UL
		if(((cell & board->kings) << 4) & empty)
		{
			out[0] = GPUBitBoard((board->white & ~cell) | (cell << 4), board->black, (board->kings & ~cell) | (cell << 4));
		}
		//UR
		if((((cell & board->kings) & GPUBitBoard::L5Mask) << 5) & empty)
		{
			out[1] = GPUBitBoard((board->white & ~cell) | (cell << 5), board->black, (board->kings & ~cell) | (cell << 5));
		}
		//LL
		if((cell >> 4) & empty)
		{
			out[2] = GPUBitBoard((board->white & ~cell) | (cell >> 4), board->black, (board->kings & ~cell) | (((board->kings & cell) >> 4) | ((cell >> 4) & GPUBitBoard::WhiteKingMask)));
		}
		//LR
		if(((cell & GPUBitBoard::R3Mask) >> 3 )& empty)
		{
			out[3] = GPUBitBoard((board->white & ~cell) | (cell >> 3), board->black, (board->kings & ~cell) | (((board->kings & cell) >> 3) | ((cell >> 3) & GPUBitBoard::WhiteKingMask)));
		}
	}
	else
	{
		//UL
		if((((cell & GPUBitBoard::L3Mask) & board->kings) << 3 ) & empty)
		{
			out[0] = GPUBitBoard((board->white & ~cell) | (cell << 3), board->black, (board->kings & ~cell) | (cell << 3));
		}
		//UR
		if(((cell & board->kings) << 4) & empty)
		{
			out[1] = GPUBitBoard((board->white & ~cell) | (cell << 4), board->black, (board->kings & ~cell) | (cell << 4));
		}
		//LL
		if(((cell & GPUBitBoard::R5Mask) >> 5) & empty)
		{
			out[2] = GPUBitBoard((board->white & ~cell) | (cell >> 5), board->black, (board->kings & ~cell) | (((board->kings & cell) >> 5) | ((cell >> 5) & GPUBitBoard::WhiteKingMask)));
		}
		//LR
		if((cell >> 4) & empty)
		{
			out[3] = GPUBitBoard((board->white & ~cell) | (cell >> 4), board->black, (board->kings & ~cell) | (((board->kings & cell) >> 4) | ((cell >> 4) & GPUBitBoard::WhiteKingMask)));
		}
	}

}

__device__ void GenWhiteJump(GPUBitBoard::board_type cell, GPUBitBoard *out, GPUBitBoard const *board)
{
	
	GPUBitBoard::board_type empty = ~(board->white | board->black);
	if(oddrow & cell)
	{
		//UL
		if(((cell & board->kings) << 4) & board->black)
		{
			GPUBitBoard::board_type j = cell << 4;
			if(((j & GPUBitBoard::L3Mask) << 3) & empty)
				out[0] = GPUBitBoard((board->white & ~cell) | (j << 3), (board->black & ~j), (board->kings & (~cell ^ j)) | (j << 3));
		}
		//UR
		if((((cell & board->kings) & GPUBitBoard::L5Mask) << 5) & board->black)
		{
			GPUBitBoard::board_type j = cell << 5;
			if((j << 4) & empty)
				out[1] = GPUBitBoard((board->white & ~cell) | (j << 4), (board->black & ~j), (board->kings & (~cell ^ j)) | (j << 4));
		}
		//LL
		if((cell >> 4) & board->black)
		{
			GPUBitBoard::board_type j = cell >> 4;
			if(((j & GPUBitBoard::R5Mask) >> 5) & empty)
				out[2] = GPUBitBoard((board->white & ~cell) | (j >> 5), (board->black & ~j), (board->kings & (~cell ^ j)) | ((((board->kings & cell) >> 4) >> 5) | ((j >> 5) & GPUBitBoard::WhiteKingMask)));
		}
		//LR
		if(((cell & GPUBitBoard::R3Mask) >> 3 )& board->black)
		{
			GPUBitBoard::board_type j = cell >> 3;
			if((j >> 4) & empty)
				out[3] = GPUBitBoard((board->white & ~cell) | (j >> 4), (board->black & ~j), (board->kings & (~cell ^ j)) | ((((board->kings & cell) >> 3) >> 4) | ((j >> 4) & GPUBitBoard::WhiteKingMask)));
		}
	}
	else
	{
		//UL
		if((((cell & GPUBitBoard::L3Mask) & board->kings) << 3 ) & board->black)
		{
			
			GPUBitBoard::board_type j = cell << 3;
			if((j << 4) & empty)
				out[0] = GPUBitBoard((board->white & ~cell) | (j << 4), (board->black & ~j), (board->kings & (~cell ^ j)) | (j << 4));
		}
		//UR
		if(((cell & board->kings) << 4) & board->black)
		{
			GPUBitBoard::board_type j = cell << 4;
			if(((j & GPUBitBoard::L5Mask) << 5) & empty)
				out[1] = GPUBitBoard((board->white & ~cell) | (j << 5), (board->black & ~j), (board->kings & (~cell ^ j)) | (j << 5));
		}
		//LL
		if(((cell & GPUBitBoard::R5Mask) >> 5) & board->black)
		{
			GPUBitBoard::board_type j = cell >> 5;
			if((j >> 4) & empty)
				out[2] = GPUBitBoard((board->white & ~cell) | (j >> 4), (board->black & ~j), (board->kings & (~cell ^ j)) | ((((board->kings & cell) >> 5) >> 4) | ((j >> 4) & GPUBitBoard::WhiteKingMask)));
		}
		//LR
		if((cell >> 4) & board->black)
		{
			GPUBitBoard::board_type j = cell >>4;
			if(((j & GPUBitBoard::R3Mask) >> 3) & empty)
				out[3] = GPUBitBoard((board->white & ~cell) | (j >> 3), (board->black & ~j), (board->kings & (~cell ^ j)) | ((((board->kings & cell) >> 4) >> 3) | ((j >> 3) & GPUBitBoard::WhiteKingMask)));
		}
	}
}

__device__ void GenBlackMove(GPUBitBoard::board_type cell, GPUBitBoard *out, GPUBitBoard const *board)
{
	// GPUBitBoard must have a validity boolean.
	
	GPUBitBoard::board_type empty = ~(board->white | board->black);

	if(oddrow & cell)
	{
		//UL
		if((cell << 4) & empty)
		{
			out[0] = GPUBitBoard(board->white, (board->black & ~cell) | (cell << 4), (board->kings & ~cell) | (((board->kings & cell) << 4) |((cell << 4) & GPUBitBoard::BlackKingMask)));
		}
		//UR
		if(((cell & GPUBitBoard::L5Mask) << 5) & empty)
		{
			out[1] = GPUBitBoard(board->white, (board->black & ~cell) | (cell << 5), (board->kings & ~cell) | (((board->kings & cell) << 5) |((cell << 5) & GPUBitBoard::BlackKingMask)));
		}
		//LL
		if(((cell & board->kings) >> 4) & empty)
		{
			out[2] = GPUBitBoard(board->white, (board->black & ~cell) | (cell >> 4), (board->kings & ~cell) | (cell >> 4));
		}
		//LR
		if((((cell & GPUBitBoard::R3Mask)& board->kings) >> 3 )& empty)
		{
			out[3] = GPUBitBoard(board->white, (board->black & ~cell) | (cell >> 3), (board->kings & ~cell) | (cell >> 3));
		}
	}
	else
	{
		//UL
		if(((cell & GPUBitBoard::L3Mask) << 3 ) & empty)
		{
			out[0] = GPUBitBoard(board->white, (board->black & ~cell) | (cell << 3), (board->kings & ~cell) | (((board->kings & cell) << 3) | ((cell <<3) & GPUBitBoard::BlackKingMask)));
		}
		//UR
		if((cell << 4) & empty)
		{
			out[1] = GPUBitBoard(board->white, (board->black & ~cell) | (cell << 4), (board->kings & ~cell) | (((board->kings & cell) << 4) | ((cell << 4) & GPUBitBoard::BlackKingMask)));
		}
		//LL
		if((((cell & board->kings) & GPUBitBoard::R5Mask) >> 5) & empty)
		{
			out[2] = GPUBitBoard(board->white, (board->black & ~cell) | (cell >> 5), (board->kings & ~cell) | (cell >> 5));
		}
		//LR
		if(((cell & board->kings) >> 4) & empty)
		{
			out[3] = GPUBitBoard(board->white, (board->black & ~cell) | (cell >> 4), (board->kings & ~cell) | (cell >> 4));
		}
	}		
}

__device__ void GenBlackJump(GPUBitBoard::board_type cell, GPUBitBoard *out, GPUBitBoard const *board)
{
	// GPUBitBoard must have a validity boolean.
	
	GPUBitBoard::board_type empty = ~(board->white | board->black);

	if(oddrow & cell)
	{
		//UL
		if((cell << 4) & board->white)
		{
			GPUBitBoard::board_type j = cell << 4;
			if(((j & GPUBitBoard::L3Mask) << 3) & empty)
				out[0] = GPUBitBoard((board->white & ~j), (board->black & ~cell) | (j << 3), (board->kings & (~cell ^ j)) | ((((board->kings & cell) << 4) << 3) | ((j << 3) & GPUBitBoard::BlackKingMask)));
		}
		//UR
		if(((cell & GPUBitBoard::L5Mask) << 5) & board->white)
		{
			GPUBitBoard::board_type j = cell << 5;
			if((j << 4) & empty)
				out[1] = GPUBitBoard((board->white & ~j), (board->black & ~cell) | (j << 4), (board->kings & (~cell ^ j)) | ((((board->kings & cell) << 5) << 4) | ((j << 4) & GPUBitBoard::BlackKingMask)));
		}
		//LL
		if(((cell & board->kings) >> 4) & board->white)
		{
			GPUBitBoard::board_type j = cell >> 4;
			if(((j & GPUBitBoard::R5Mask) >> 5) & empty)
				out[2] = GPUBitBoard((board->white & ~j), (board->black & ~cell) | (j >> 5), (board->kings & (~cell ^ j)) | (j >> 5));
		}
		//LR
		if((((cell & GPUBitBoard::R3Mask) & board->kings) >> 3) & board->white)
		{
			GPUBitBoard::board_type j = cell >> 3;
			if((j >> 4) & empty)
				out[3] = GPUBitBoard((board->white & ~j), (board->black & ~cell) | (j >> 4), (board->kings & (~cell ^ j)) | (j >> 4));
		}
	}
	else
	{
		//UL
		if(((cell & GPUBitBoard::L3Mask) << 3 ) & board->white)
		{
			GPUBitBoard::board_type j = cell << 3;
			if((j << 4) & empty)
				out[0] = GPUBitBoard((board->white & ~j), (board->black & ~cell) | (j << 4), (board->kings & (~cell ^ j)) | ((((board->kings & cell) << 3) << 4) | ((j << 4) & GPUBitBoard::BlackKingMask)));
		}
		//UR
		if((cell << 4) & board->white)
		{
			GPUBitBoard::board_type j = cell << 4;
			if(((j & GPUBitBoard::L5Mask) << 5) & empty)
				out[1] = GPUBitBoard((board->white & ~j), (board->black & ~cell) | (j << 5), (board->kings & (~cell ^ j)) | ((((board->kings & cell) << 4) << 5) | ((j << 5) & GPUBitBoard::BlackKingMask)));
		}
		//LL
		if((((cell & board->kings) & GPUBitBoard::R5Mask) >> 5) & board->white)
		{
			GPUBitBoard::board_type j = cell >> 5;
			if((j >> 4) & empty)
				out[2] = GPUBitBoard((board->white & ~j), (board->black & ~cell) | (j >> 4), (board->kings & (~cell ^ j)) | (j >> 4));
		}
		//LR
		if(((cell & board->kings) >> 4) & board->white)
		{
			GPUBitBoard::board_type j = cell >> 4;
			if(((j & GPUBitBoard::R3Mask) >> 3) & empty)
				out[3] = GPUBitBoard((board->white & ~j), (board->black & ~cell) | (j >> 3), (board->kings & (~cell ^ j)) | (j >> 3));
		}
	}
		
}

using void(*)*GPUBitBoard::board_type, GPUBitBoard *, GPUBitBoard const *) = gen_move_func;
__device__ gen_white_move[2] = { GenWhiteMove, GenWhiteJump };
__device__ gen_black_move[2] = { GenBlackMove, GenBlackJump };


// will need master_black_max_kernel, master_black_min_kernel, black_max_kernel and black_min_kernel as well.
// non_master kernels will create 4 streams to use on its own per thread.

__global__ void master_white_max_kernel(utility_type *v, utility_type *utility, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns)
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
			alpha = max(alpha, *v);
		}
		
	}
	
	__syncthreads();
}

__global__ master_white_min_kernel(utility_type *v, utility_type *utility, GPUBitBoard const *src, int num_boards, int alpha, int beta, int depth, int turns)
{
	// same logic as master_white_max_kernel, but use min node logic
}

__global__ white_min_kernel(utility_type *v, GPUBitBoard const *src, int alpha, int beta, int depth, int turns)
{
	int tx = threadIdx.x;
	int t_beta = beta;
	__shared__ bool terminated;
	__shared__ utility_type utilities[32];

	if(tx < 32)
	{
		utilities[tx] = *v;
	}
	
	__syncthreads();

	if(tx == 0)
	{
		utility_type terminal_value = 0;
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
		utility_type *utility = cudaMalloc(sizeof(utility_type) * 4);
		utility[0] = utility[1] = utility[2] = utility[3] = utilities[tx];
		GPUBitBoard *new_boards = cudaMalloc(sizeof(GPUBitBoard) * 4);
		
		cudaStream_t streams[4];
		cudaStreamCreateWithFlags(&streams[0], cudaStreamNonBlocking);
		cudaStreamCreateWithFlags(&streams[1], cudaStreamNonBlocking);
		cudaStreamCreateWithFlags(&streams[2], cudaStreamNonBlocking);
		cudaStreamCreateWithFlags(&streams[3], cudaStreamNonBlocking);
		
		// in the max kernel, use gen_black_move_type instead
		int gen_black_move_type = (int)GPUBitBoard::GetBlackJumps(src) != 0);
		gen_black_move[gen_black_move_type](1u << tx, new_board, src);
		
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

__global__ white_max_kernel(utility_type *v, GPUBitBoard const *src, int alpha, int beta, int depth, int turns)
{
	// same as white_min_kernel, but use max logic.
}