
// modifications to old structures:
// GPUBitBoard structure should no longer have valid flag
// GenWhiteMove/GenWhiteJump/GenMoreWhiteJumps and GenBlackMove/GenBlackJump/GenMoreBlackJumps now must have this signature:

// atomic version should be device only, and non-atomic version should be both host and device capable.

// __host__ __device__ void GPUBitBoard::GenWhiteMove(GPUBitBoard::board_type cell, GPUBitBoard const &board, GPUBitBoard *frontier, int &frontier_size);
// __device__ void GPUBitBoard::GenWhiteMoveAtomic(GPUBitBoard::board_type cell, GPUBitBoard const &board, GPUBitBoard *frontier, int *frontier_size);

// for non-atomic version of gen_move, can just perform frontier[f_size++] = /* new board */.
// for atomic version of gen_move, must perform frontier[atomicAdd(f_size, 1)] = /* new board */.

__device__ utility_type explore_black_frontier(GPUBitBoard const &board, utility_type alpha, utility_type beta, /*minimax node enum */ node_type, int depth, int turns)
{
	GPUBitBoard frontier[32];
	int frontier_size = 0;
	int v = (node_type == max) ? -Infinity : Infinity;
	
	int gen_board_type;
	
	utility_type terminal_value = 0;
	if(GetWhiteUtility(board, terminal_value, depth, turns))
	{
		return terminal_value;
	}
	
	if(node_type == max)
	{
		gen_board_type = (GPUBitBoard::GetBlackJumps(board[bx]) != 0) ? 1 : 0;
	}
	else
	{
		gen_board_type = (GPUBitBoard::GetWhiteJumps(board[bx]) != 0) ? 1 : 0;
	}
	
	if(node_type == max)
	{
		// if dynamic parallelism is possible, can call another kernel here
		for(int i = 0; i < 32; ++i)
		{
			gen_black_move_func[gen_board_type](1u << i, board[tx], frontier, &f_size);
		}
		
		while(f_size > 0)
		{
			v = max(explore_black_frontier(frontier[--f_size], alpha, beta, node_type + 1), v);
			if(v > beta)
			{
				break;
			}
			alpha = max(alpha, v);
		}
	}
	else
	{
		// if dynamic parallelism is possible, can call another kernel here
		for(int i = 0; i < 32; ++i)
		{
			gen_white_move_func[gen_board_type](1u << i, board[tx], frontier, &f_size);
		}
		
		while(f_size > 0)
		{
			while(f_size > 0)
			{
				v = min(explore_black_frontier(frontier[--f_size], alpha, beta, node_type + 1, v);
				if(v < alpha)
				{
					break;
				}
				beta = min(beta, v);
			}
		}
	}
	
	return v;
}

// v -> num_board * sizeof(utility_type), host takes v[0].
// host will launch kernel by doing black_kernel<<<dim3(num_boards, 1, 1), dim3(32, 1, 1)>>>(...);

__global__ void black_kernel(utility_type *v, utility_type X, GPUBitBoard const *boards, int num_boards, utility_type alpha, utility_type beta, /* minimax node enum */ node_type, int depth, int turns)
{
	int tx = threadIdx.x;
	int bx = blockIdx.x;
	
	__shared__ int f_size;
	__shared__ int gen_board_type;
	__shared__ GPUBitBoard frontier[32];
	__shared__ utility_type t_v[32];
	
	utility_type cell = 1u << tx;
	
	if(tx < 32)
	{
		t_v[tx] = *v;
		if(tx == 0)
		{
			f_size = 0;
			if(node_type == max)
			{
				gen_board_type = (GPUBitBoard::GetBlackJumps(board[bx]) != 0) ? 1 : 0;
			}
			else
			{
				gen_board_type = (GPUBitBoard::GetWhiteJumps(board[bx]) != 0) ? 1 : 0;
			}
		}
	}
	
	__syncthreads();
	
	if(node_type == max)
	{
		
		gen_black_move_atomic_func[gen_board_type](1u << tx, board[tx], frontier, &f_size);
	}
	else
	{
		gen_white_move_atomic_func[gen_board_type](1u << tx, board[tx], frontier, &f_size);
	}
	
	__syncthreads();
	
	
	if(tx < f_size)
	{
		t_v[tx] = explore_black_frontier(frontier[tx], alpha, beta, node_type + 1, depth - 1, turns - 1);
	}
	
	__syncthreads();
	
	if(tx == 0)
	{
		// ab-prune t_v and send the last value to v[bx].
		if(node_type == max)
		{
			while(f_size > 0)
			{
				X = max(t_v[--f_size], X);
				if(X > beta)
				{
					break;
				}
				alpha = max(alpha, X);
			}
		}
		else
		{
			while(f_size > 0)
			{
				X = min(t_v[--f_size], X);
				if(X < alpha)
				{
					break;
				}
				beta = min(beta, X);
			}
		}
		
		v[bx] = X;
	}
	
	__syncthreads();
	if(bx == 0 && tx == 0)
	{
		X = v[0];
		// ab-prune v and send the last value to v[0].
		if(node_type == max)
		{
			for(int i = 1; i < num_boards; ++i)
			{
				X = max(v[i], X);
				if(v[0] > beta)
				{
					break;
				}
				alpha = max(alpha, X);
			}
		}
		else
		{
			for(int i = 1; i < num_boards; ++i)
			{
				X = min(v[i], X);
				if(v[0] < alpha)
				{
					break;
				}
				beta = min(beta, X);
			}
		}
		
		v[0] = X;
	}
	
	__syncthreads();
}