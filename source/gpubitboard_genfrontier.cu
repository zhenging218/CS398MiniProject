#include <helper_cuda.h>
#include "gpubitboard.cuh"

namespace Checkers
{
	// frontier generation
	// each block has 32 threads, 1 cell per thread
	// each block processes 1 board, grid size should be no. of boards

	/*
	__device__ BitBoard::board_type GetWhiteMoves(BitBoard const *b)
	{
		const BitBoard::board_type not_occupied = ~(b->black | b->white);
		const BitBoard::board_type white_kings = b->white & b->kings;
		BitBoard::board_type LL = (((not_occupied & BitBoard::R5Mask) >> 5) & white_kings) | ((not_occupied >> 4) & white_kings);
		BitBoard::board_type LR = (((not_occupied) >> 4) & white_kings) | (((not_occupied & BitBoard::R3Mask) >> 3) & white_kings);
		BitBoard::board_type UL = (((not_occupied & BitBoard::L5Mask) << 5) & b->white) | ((not_occupied << 4) & b->white);
		BitBoard::board_type UR = (((not_occupied << 4) & b->white) | ((not_occupied & BitBoard::L3Mask) << 3) & b->white);

		return  (LL | LR | UL | UR);
	}

	__device__ BitBoard::board_type GetBlackMoves(BitBoard const *b)
	{
		const BitBoard::board_type not_occupied = ~(b->black | b->white);
		const BitBoard::board_type black_kings = b->black & b->kings;
		BitBoard::board_type LL = (((not_occupied & BitBoard::R5Mask) >> 5) & b->black) | ((not_occupied >> 4) & b->black);
		BitBoard::board_type LR = (((not_occupied) >> 4) & b->black) | (((not_occupied & BitBoard::R3Mask) >> 3) & b->black);
		BitBoard::board_type UL = (((not_occupied & BitBoard::L5Mask) << 5) & black_kings) | ((not_occupied << 4) & black_kings);
		BitBoard::board_type UR = (((not_occupied << 4) & black_kings) | ((not_occupied & BitBoard::L3Mask) << 3) & black_kings);

		return  (LL | LR | UL | UR);
	}

	__device__ void GenerateWhiteFrontierUL(BitBoard const *d_DataIn, BitBoard *d_DataOut)
	{
		std::uint32_t tx = threadIdx.x;
		std::uint32_t index = blockIdx.x * blockDim.x + tx;
		BitBoard::board_type moves = GetWhiteMoves(d_DataIn + blockIdx.x);

		if (tx < (sizeof(std::uint32_t) * 8))
		{
			// get bitshift
			BitBoard::board_type i = 1u << tx;
			if (BitBoard::OddRows & i)
			{
				// gen frontier for this board
				d_DataOut[tx] = BitBoard()
			}
			else
			{
				// gen frontier for this board
			}
		}
	}

	__device__ GenerateWhiteFrontierUR(BitBoard const *d_DataIn, BitBoard *d_DataOut)
	{

	}

	__device__ GenerateWhiteFrontierLL(BitBoard const *d_DataIn, BitBoard *d_DataOut)
	{

	}

	__device__ GenerateWhiteFrontierLR(BitBoard const *d_DataIn, BitBoard *d_DataOut)
	{

	}
	*/
}