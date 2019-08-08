#include "precomp.cuh"

namespace Checkers
{
	__host__ __device__ GPUBitBoard::GPUBitBoard() : white(0xFFF00000u), black(0x00000FFFu), kings(0u), valid(false)
	{

	}

	__host__ __device__ GPUBitBoard::GPUBitBoard(board_type w, board_type b, board_type k) : white(w), black(b), kings(k), valid(true)
	{

	}

	__host__ __device__ GPUBitBoard &GPUBitBoard::operator=(GPUBitBoard const &src)
	{
		white = src.white;
		black = src.black;
		kings = src.kings;
		valid = src.valid;
		return *this;
	}

	__host__ GPUBitBoard::GPUBitBoard(BitBoard const &src) : white(src.white), black(src.black), kings(src.kings)
	{

	}

	__host__ GPUBitBoard &GPUBitBoard::operator=(BitBoard const &src)
	{
		white = src.white;
		black = src.black;
		kings = src.kings;
		valid = true;
		return *this;
	}

	__host__ GPUBitBoard::operator BitBoard() const
	{
		return BitBoard(white, black, kings);
	}

	__host__ __device__ GPUBitBoard::board_type GPUBitBoard::GetBlackMoves(GPUBitBoard const &b)
	{
		const board_type not_occupied = ~(b.black | b.white);
		const board_type black_kings = b.black & b.kings;
		board_type LL = (((not_occupied & R5Mask) >> 5) & b.black) | ((not_occupied >> 4) & b.black);
		board_type LR = (((not_occupied) >> 4) & b.black) | (((not_occupied & R3Mask) >> 3) & b.black);
		board_type UL = (((not_occupied & L5Mask) << 5) & black_kings) | ((not_occupied << 4) & black_kings);
		board_type UR = (((not_occupied << 4) & black_kings) | ((not_occupied & L3Mask) << 3) & black_kings);

		return  (LL | LR | UL | UR);
	}

	__host__ __device__ GPUBitBoard::board_type GPUBitBoard::GetWhiteMoves(GPUBitBoard const &b)
	{
		const board_type not_occupied = ~(b.black | b.white);
		const board_type white_kings = b.white & b.kings;
		board_type LL = (((not_occupied & R5Mask) >> 5) & white_kings) | ((not_occupied >> 4) & white_kings);
		board_type LR = (((not_occupied) >> 4) & white_kings) | (((not_occupied & R3Mask) >> 3) & white_kings);
		board_type UL = (((not_occupied & L5Mask) << 5) & b.white) | ((not_occupied << 4) & b.white);
		board_type UR = (((not_occupied << 4) & b.white) | ((not_occupied & L3Mask) << 3) & b.white);

		return  (LL | LR | UL | UR);
	}

	__host__ __device__ GPUBitBoard::board_type GPUBitBoard::GetBlackJumps(GPUBitBoard const &b)
	{
		const board_type not_occupied = ~(b.black | b.white);
		const board_type black_kings = b.black & b.kings;
		board_type LL_even_to_odd = (((((not_occupied & R5Mask) >> 5) & b.white) >> 4) & b.black);
		board_type LR_even_to_odd = (((((not_occupied >> 4) & b.white) & R3Mask) >> 3) & b.black);
		board_type UL_even_to_odd = (((((not_occupied & L5Mask) << 5) & b.white) << 4) & black_kings);
		board_type UR_even_to_odd = (((((not_occupied << 4) & b.white) & L3Mask) << 3) & black_kings);

		board_type LL_odd_to_even = (((((not_occupied >> 4) & b.white) & R5Mask) >> 5) & b.black);
		board_type LR_odd_to_even = (((((not_occupied & R3Mask) >> 3) & b.white) >> 4) & b.black);
		board_type UL_odd_to_even = (((((not_occupied << 4) & b.white) & L5Mask) << 5) & black_kings);
		board_type UR_odd_to_even = (((((not_occupied & L3Mask) << 3) & b.white) << 4) & black_kings);

		board_type move = (LL_even_to_odd | LR_even_to_odd | UL_even_to_odd | UR_even_to_odd |
			LL_odd_to_even | LR_odd_to_even | UL_odd_to_even | UR_odd_to_even);
		return move;
	}

	__host__ __device__ GPUBitBoard::board_type GPUBitBoard::GetWhiteJumps(GPUBitBoard const &b)
	{
		const board_type not_occupied = ~(b.black | b.white);
		const board_type white_kings = b.white & b.kings;
		board_type LL_even_to_odd = (((((not_occupied & R5Mask) >> 5) & b.black) >> 4) & white_kings);
		board_type LR_even_to_odd = (((((not_occupied >> 4) & b.black) & R3Mask) >> 3) & white_kings);
		board_type UL_even_to_odd = (((((not_occupied & L5Mask) << 5) & b.black) << 4) & b.white);
		board_type UR_even_to_odd = (((((not_occupied << 4) & b.black) & L3Mask) << 3) & b.white);

		board_type LL_odd_to_even = (((((not_occupied >> 4) & b.black) & R5Mask) >> 5) & white_kings);
		board_type LR_odd_to_even = (((((not_occupied & R3Mask) >> 3) & b.black) >> 4) & white_kings);
		board_type UL_odd_to_even = (((((not_occupied << 4) & b.black) & L5Mask) << 5) & b.white);
		board_type UR_odd_to_even = (((((not_occupied & L3Mask) << 3) & b.black) << 4) & b.white);

		board_type move = (LL_even_to_odd | LR_even_to_odd | UL_even_to_odd | UR_even_to_odd |
			LL_odd_to_even | LR_odd_to_even | UL_odd_to_even | UR_odd_to_even);
		return move;
	}

	__host__ __device__ GPUBitBoard::count_type GPUBitBoard::GetBlackPieceCount(GPUBitBoard const &b)
	{
		return GPUSWAR32(b.black);
	}

	__host__ __device__ GPUBitBoard::count_type GPUBitBoard::GetWhitePieceCount(GPUBitBoard const &b)
	{
		return GPUSWAR32(b.white);
	}

	__host__ __device__ GPUBitBoard::count_type GPUBitBoard::GetBlackKingsCount(GPUBitBoard const &b)
	{
		return GPUSWAR32(b.black & b.kings);
	}

	__host__ __device__ GPUBitBoard::count_type GPUBitBoard::GetWhiteKingsCount(GPUBitBoard const &b)
	{
		return GPUSWAR32(b.white & b.kings);
	}

	__host__ __device__ void GPUBitBoard::GenMoreWhiteJumps(GPUBitBoard::board_type cell, GPUBitBoard *&out, GPUBitBoard const &board)
	{
		GPUBitBoard::board_type jumps = GPUBitBoard::GetWhiteJumps(board);
		GPUBitBoard::board_type empty = ~(board.white | board.black);

		if (jumps & cell)
		{
			if (OddRows & cell)
			{
				//UL
				if (((cell & board.kings) << 4) & board.black)
				{
					GPUBitBoard::board_type j = cell << 4;
					if (((j & GPUBitBoard::L3Mask) << 3) & empty)
						GenMoreWhiteJumps(j << 3, out, GPUBitBoard((board.white & ~cell) | (j << 3), (board.black & ~j), (board.kings & (~cell ^ j)) | (j << 3)));
				}
				//UR
				if ((((cell & board.kings) & GPUBitBoard::L5Mask) << 5) & board.black)
				{
					GPUBitBoard::board_type j = cell << 5;
					if ((j << 4) & empty)
						GenMoreWhiteJumps(j << 4, out, GPUBitBoard((board.white & ~cell) | (j << 4), (board.black & ~j), (board.kings & (~cell ^ j)) | (j << 4)));
				}
				//LL
				if ((cell >> 4) & board.black)
				{
					GPUBitBoard::board_type j = cell >> 4;
					if (((j & GPUBitBoard::R5Mask) >> 5) & empty)
						GenMoreWhiteJumps(j >> 5, out, GPUBitBoard((board.white & ~cell) | (j >> 5), (board.black & ~j), (board.kings & (~cell ^ j)) | ((((board.kings & cell) >> 4) >> 5) | ((j >> 5) & GPUBitBoard::WhiteKingMask))));
				}
				//LR
				if (((cell & GPUBitBoard::R3Mask) >> 3)& board.black)
				{
					GPUBitBoard::board_type j = cell >> 3;
					if ((j >> 4) & empty)
						GenMoreWhiteJumps(j >> 4, out, GPUBitBoard((board.white & ~cell) | (j >> 4), (board.black & ~j), (board.kings & (~cell ^ j)) | ((((board.kings & cell) >> 3) >> 4) | ((j >> 4) & GPUBitBoard::WhiteKingMask))));
				}
			}
			else
			{
				//UL
				if ((((cell & GPUBitBoard::L3Mask) & board.kings) << 3) & board.black)
				{

					GPUBitBoard::board_type j = cell << 3;
					if ((j << 4) & empty)
						GenMoreWhiteJumps(j << 4, out, GPUBitBoard((board.white & ~cell) | (j << 4), (board.black & ~j), (board.kings & (~cell ^ j)) | (j << 4)));
				}
				//UR
				if (((cell & board.kings) << 4) & board.black)
				{
					GPUBitBoard::board_type j = cell << 4;
					if (((j & GPUBitBoard::L5Mask) << 5) & empty)
						GenMoreWhiteJumps(j << 5, out, GPUBitBoard((board.white & ~cell) | (j << 5), (board.black & ~j), (board.kings & (~cell ^ j)) | (j << 5)));
				}
				//LL
				if (((cell & GPUBitBoard::R5Mask) >> 5) & board.black)
				{
					GPUBitBoard::board_type j = cell >> 5;
					if ((j >> 4) & empty)
						GenMoreWhiteJumps(j >> 4, out, GPUBitBoard((board.white & ~cell) | (j >> 4), (board.black & ~j), (board.kings & (~cell ^ j)) | ((((board.kings & cell) >> 5) >> 4) | ((j >> 4) & GPUBitBoard::WhiteKingMask))));
				}
				//LR
				if ((cell >> 4) & board.black)
				{
					GPUBitBoard::board_type j = cell >> 4;
					if (((j & GPUBitBoard::R3Mask) >> 3) & empty)
						GenMoreWhiteJumps(j >> 3, out, GPUBitBoard((board.white & ~cell) | (j >> 3), (board.black & ~j), (board.kings & (~cell ^ j)) | ((((board.kings & cell) >> 4) >> 3) | ((j >> 3) & GPUBitBoard::WhiteKingMask))));
				}
			}
		}
		else
		{
			*(out++) = board;
		}
	}

	__host__ __device__ void GPUBitBoard::GenMoreBlackJumps(GPUBitBoard::board_type cell, GPUBitBoard *&out, GPUBitBoard const &board)
	{
		GPUBitBoard::board_type jumps = GPUBitBoard::GetBlackJumps(board);
		GPUBitBoard::board_type empty = ~(board.white | board.black);

		if (jumps & cell)
		{
			if (OddRows & cell)
			{
				//UL
				if ((cell << 4) & board.white)
				{
					GPUBitBoard::board_type j = cell << 4;
					if (((j & GPUBitBoard::L3Mask) << 3) & empty)
						GenMoreBlackJumps(j << 3, out, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j << 3), (board.kings & (~cell ^ j)) | ((((board.kings & cell) << 4) << 3) | ((j << 3) & GPUBitBoard::BlackKingMask))));
				}
				//UR
				if (((cell & GPUBitBoard::L5Mask) << 5) & board.white)
				{
					GPUBitBoard::board_type j = cell << 5;
					if ((j << 4) & empty)
						GenMoreBlackJumps(j << 4, out, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j << 4), (board.kings & (~cell ^ j)) | ((((board.kings & cell) << 5) << 4) | ((j << 4) & GPUBitBoard::BlackKingMask))));
				}
				//LL
				if (((cell & board.kings) >> 4) & board.white)
				{
					GPUBitBoard::board_type j = cell >> 4;
					if (((j & GPUBitBoard::R5Mask) >> 5) & empty)
						GenMoreBlackJumps(j >> 5, out, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j >> 5), (board.kings & (~cell ^ j)) | (j >> 5)));
				}
				//LR
				if ((((cell & GPUBitBoard::R3Mask) & board.kings) >> 3) & board.white)
				{
					GPUBitBoard::board_type j = cell >> 3;
					if ((j >> 4) & empty)
						GenMoreBlackJumps(j >> 4, out, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j >> 4), (board.kings & (~cell ^ j)) | (j >> 4)));
				}
			}
			else
			{
				//UL
				if (((cell & GPUBitBoard::L3Mask) << 3) & board.white)
				{
					GPUBitBoard::board_type j = cell << 3;
					if ((j << 4) & empty)
						GenMoreBlackJumps(j << 4, out, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j << 4), (board.kings & (~cell ^ j)) | ((((board.kings & cell) << 3) << 4) | ((j << 4) & GPUBitBoard::BlackKingMask))));
				}
				//UR
				if ((cell << 4) & board.white)
				{
					GPUBitBoard::board_type j = cell << 4;
					if (((j & GPUBitBoard::L5Mask) << 5) & empty)
						GenMoreBlackJumps(j << 5, out, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j << 5), (board.kings & (~cell ^ j)) | ((((board.kings & cell) << 4) << 5) | ((j << 5) & GPUBitBoard::BlackKingMask))));
				}
				//LL
				if ((((cell & board.kings) & GPUBitBoard::R5Mask) >> 5) & board.white)
				{
					GPUBitBoard::board_type j = cell >> 5;
					if ((j >> 4) & empty)
						GenMoreBlackJumps(j >> 4, out, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j >> 4), (board.kings & (~cell ^ j)) | (j >> 4)));
				}
				//LR
				if (((cell & board.kings) >> 4) & board.white)
				{
					GPUBitBoard::board_type j = cell >> 4;
					if (((j & GPUBitBoard::R3Mask) >> 3) & empty)
						GenMoreBlackJumps(j >> 3, out, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j >> 3), (board.kings & (~cell ^ j)) | (j >> 3)));
				}
			}
		}
		else
		{
			*(out++) = board;
		}
	}

	__host__ __device__ void GPUBitBoard::GenWhiteMove(GPUBitBoard::board_type cell, GPUBitBoard *&out, GPUBitBoard const &board)
	{
		// GPUBitBoard must have a validity boolean.
		//normal move (UL, UR)
		//king extra moves (LL, LR)
		GPUBitBoard::board_type empty = ~(board.white | board.black);

		if (GPUBitBoard::OddRows & cell)
		{
			//UL
			if (((cell & board.kings) << 4) & empty)
			{
				(*out++) = GPUBitBoard((board.white & ~cell) | (cell << 4), board.black, (board.kings & ~cell) | (cell << 4));
			}
			//UR
			if ((((cell & board.kings) & GPUBitBoard::L5Mask) << 5) & empty)
			{
				*(out++) = GPUBitBoard((board.white & ~cell) | (cell << 5), board.black, (board.kings & ~cell) | (cell << 5));
			}
			//LL
			if ((cell >> 4) & empty)
			{
				*(out++) = GPUBitBoard((board.white & ~cell) | (cell >> 4), board.black, (board.kings & ~cell) | (((board.kings & cell) >> 4) | ((cell >> 4) & GPUBitBoard::WhiteKingMask)));
			}
			//LR
			if (((cell & GPUBitBoard::R3Mask) >> 3)& empty)
			{
				*(out++) = GPUBitBoard((board.white & ~cell) | (cell >> 3), board.black, (board.kings & ~cell) | (((board.kings & cell) >> 3) | ((cell >> 3) & GPUBitBoard::WhiteKingMask)));
			}
		}
		else
		{
			//UL
			if ((((cell & GPUBitBoard::L3Mask) & board.kings) << 3) & empty)
			{
				*(out++) = GPUBitBoard((board.white & ~cell) | (cell << 3), board.black, (board.kings & ~cell) | (cell << 3));
			}
			//UR
			if (((cell & board.kings) << 4) & empty)
			{
				*(out++) = GPUBitBoard((board.white & ~cell) | (cell << 4), board.black, (board.kings & ~cell) | (cell << 4));
			}
			//LL
			if (((cell & GPUBitBoard::R5Mask) >> 5) & empty)
			{
				*(out++) = GPUBitBoard((board.white & ~cell) | (cell >> 5), board.black, (board.kings & ~cell) | (((board.kings & cell) >> 5) | ((cell >> 5) & GPUBitBoard::WhiteKingMask)));
			}
			//LR
			if ((cell >> 4) & empty)
			{
				*(out++) = GPUBitBoard((board.white & ~cell) | (cell >> 4), board.black, (board.kings & ~cell) | (((board.kings & cell) >> 4) | ((cell >> 4) & GPUBitBoard::WhiteKingMask)));
			}
		}
	}

	__host__ __device__ void GPUBitBoard::GenWhiteJump(GPUBitBoard::board_type cell, GPUBitBoard *&out, GPUBitBoard const &board)
	{
		GPUBitBoard::board_type empty = ~(board.white | board.black);
		if (OddRows & cell)
		{
			//UL
			if (((cell & board.kings) << 4) & board.black)
			{
				GPUBitBoard::board_type j = cell << 4;
				if (((j & GPUBitBoard::L3Mask) << 3) & empty)
					GenMoreWhiteJumps(j << 3, out, GPUBitBoard((board.white & ~cell) | (j << 3), (board.black & ~j), (board.kings & (~cell ^ j)) | (j << 3)));
			}
			//UR
			if ((((cell & board.kings) & GPUBitBoard::L5Mask) << 5) & board.black)
			{
				GPUBitBoard::board_type j = cell << 5;
				if ((j << 4) & empty)
					GenMoreWhiteJumps(j << 4, out, GPUBitBoard((board.white & ~cell) | (j << 4), (board.black & ~j), (board.kings & (~cell ^ j)) | (j << 4)));
			}
			//LL
			if ((cell >> 4) & board.black)
			{
				GPUBitBoard::board_type j = cell >> 4;
				if (((j & GPUBitBoard::R5Mask) >> 5) & empty)
					GenMoreWhiteJumps(j >> 5, out, GPUBitBoard((board.white & ~cell) | (j >> 5), (board.black & ~j), (board.kings & (~cell ^ j)) | ((((board.kings & cell) >> 4) >> 5) | ((j >> 5) & GPUBitBoard::WhiteKingMask))));
			}
			//LR
			if (((cell & GPUBitBoard::R3Mask) >> 3)& board.black)
			{
				GPUBitBoard::board_type j = cell >> 3;
				if ((j >> 4) & empty)
					GenMoreWhiteJumps(j >> 4, out, GPUBitBoard((board.white & ~cell) | (j >> 4), (board.black & ~j), (board.kings & (~cell ^ j)) | ((((board.kings & cell) >> 3) >> 4) | ((j >> 4) & GPUBitBoard::WhiteKingMask))));
			}
		}
		else
		{
			//UL
			if ((((cell & GPUBitBoard::L3Mask) & board.kings) << 3) & board.black)
			{

				GPUBitBoard::board_type j = cell << 3;
				if ((j << 4) & empty)
					GenMoreWhiteJumps(j << 4, out, GPUBitBoard((board.white & ~cell) | (j << 4), (board.black & ~j), (board.kings & (~cell ^ j)) | (j << 4)));
			}
			//UR
			if (((cell & board.kings) << 4) & board.black)
			{
				GPUBitBoard::board_type j = cell << 4;
				if (((j & GPUBitBoard::L5Mask) << 5) & empty)
					GenMoreWhiteJumps(j << 5, out, GPUBitBoard((board.white & ~cell) | (j << 5), (board.black & ~j), (board.kings & (~cell ^ j)) | (j << 5)));
			}
			//LL
			if (((cell & GPUBitBoard::R5Mask) >> 5) & board.black)
			{
				GPUBitBoard::board_type j = cell >> 5;
				if ((j >> 4) & empty)
					GenMoreWhiteJumps(j >> 4, out, GPUBitBoard((board.white & ~cell) | (j >> 4), (board.black & ~j), (board.kings & (~cell ^ j)) | ((((board.kings & cell) >> 5) >> 4) | ((j >> 4) & GPUBitBoard::WhiteKingMask))));
			}
			//LR
			if ((cell >> 4) & board.black)
			{
				GPUBitBoard::board_type j = cell >> 4;
				if (((j & GPUBitBoard::R3Mask) >> 3) & empty)
					GenMoreWhiteJumps(j >> 3, out, GPUBitBoard((board.white & ~cell) | (j >> 3), (board.black & ~j), (board.kings & (~cell ^ j)) | ((((board.kings & cell) >> 4) >> 3) | ((j >> 3) & GPUBitBoard::WhiteKingMask))));
			}
		}
	}

	__host__ __device__ void GPUBitBoard::GenBlackMove(GPUBitBoard::board_type cell, GPUBitBoard *&out, GPUBitBoard const &board)
	{
		// GPUBitBoard must have a validity boolean.

		GPUBitBoard::board_type empty = ~(board.white | board.black);
		if (OddRows & cell)
		{
			//UL
			if ((cell << 4) & empty)
			{
				*(out++) = GPUBitBoard(board.white, (board.black & ~cell) | (cell << 4), (board.kings & ~cell) | (((board.kings & cell) << 4) | ((cell << 4) & GPUBitBoard::BlackKingMask)));
			}
			//UR
			if (((cell & GPUBitBoard::L5Mask) << 5) & empty)
			{
				*(out++) = GPUBitBoard(board.white, (board.black & ~cell) | (cell << 5), (board.kings & ~cell) | (((board.kings & cell) << 5) | ((cell << 5) & GPUBitBoard::BlackKingMask)));
			}
			//LL
			if (((cell & board.kings) >> 4) & empty)
			{
				*(out++) = GPUBitBoard(board.white, (board.black & ~cell) | (cell >> 4), (board.kings & ~cell) | (cell >> 4));
			}
			//LR
			if ((((cell & GPUBitBoard::R3Mask)& board.kings) >> 3)& empty)
			{
				*(out++) = GPUBitBoard(board.white, (board.black & ~cell) | (cell >> 3), (board.kings & ~cell) | (cell >> 3));
			}
		}
		else
		{
			//UL
			if (((cell & GPUBitBoard::L3Mask) << 3) & empty)
			{
				*(out++) = GPUBitBoard(board.white, (board.black & ~cell) | (cell << 3), (board.kings & ~cell) | (((board.kings & cell) << 3) | ((cell << 3) & GPUBitBoard::BlackKingMask)));
			}
			//UR
			if ((cell << 4) & empty)
			{
				*(out++) = GPUBitBoard(board.white, (board.black & ~cell) | (cell << 4), (board.kings & ~cell) | (((board.kings & cell) << 4) | ((cell << 4) & GPUBitBoard::BlackKingMask)));
			}
			//LL
			if ((((cell & board.kings) & GPUBitBoard::R5Mask) >> 5) & empty)
			{
				*(out++) = GPUBitBoard(board.white, (board.black & ~cell) | (cell >> 5), (board.kings & ~cell) | (cell >> 5));
			}
			//LR
			if (((cell & board.kings) >> 4) & empty)
			{
				*(out++) = GPUBitBoard(board.white, (board.black & ~cell) | (cell >> 4), (board.kings & ~cell) | (cell >> 4));
			}
		}
	}

	__host__ __device__ void GPUBitBoard::GenBlackJump(GPUBitBoard::board_type cell, GPUBitBoard *&out, GPUBitBoard const &board)
	{
		// GPUBitBoard must have a validity boolean.

		GPUBitBoard::board_type empty = ~(board.white | board.black);

		if (OddRows & cell)
		{
			//UL
			if ((cell << 4) & board.white)
			{
				GPUBitBoard::board_type j = cell << 4;
				if (((j & GPUBitBoard::L3Mask) << 3) & empty)
					GenMoreBlackJumps(j << 3, out, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j << 3), (board.kings & (~cell ^ j)) | ((((board.kings & cell) << 4) << 3) | ((j << 3) & GPUBitBoard::BlackKingMask))));
			}
			//UR
			if (((cell & GPUBitBoard::L5Mask) << 5) & board.white)
			{
				GPUBitBoard::board_type j = cell << 5;
				if ((j << 4) & empty)
					GenMoreBlackJumps(j << 4, out, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j << 4), (board.kings & (~cell ^ j)) | ((((board.kings & cell) << 5) << 4) | ((j << 4) & GPUBitBoard::BlackKingMask))));
			}
			//LL
			if (((cell & board.kings) >> 4) & board.white)
			{
				GPUBitBoard::board_type j = cell >> 4;
				if (((j & GPUBitBoard::R5Mask) >> 5) & empty)
					GenMoreBlackJumps(j >> 5, out, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j >> 5), (board.kings & (~cell ^ j)) | (j >> 5)));
			}
			//LR
			if ((((cell & GPUBitBoard::R3Mask) & board.kings) >> 3) & board.white)
			{
				GPUBitBoard::board_type j = cell >> 3;
				if ((j >> 4) & empty)
					GenMoreBlackJumps(j >> 4, out, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j >> 4), (board.kings & (~cell ^ j)) | (j >> 4)));
			}
		}
		else
		{
			//UL
			if (((cell & GPUBitBoard::L3Mask) << 3) & board.white)
			{
				GPUBitBoard::board_type j = cell << 3;
				if ((j << 4) & empty)
					GenMoreBlackJumps(j << 4, out, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j << 4), (board.kings & (~cell ^ j)) | ((((board.kings & cell) << 3) << 4) | ((j << 4) & GPUBitBoard::BlackKingMask))));
			}
			//UR
			if ((cell << 4) & board.white)
			{
				GPUBitBoard::board_type j = cell << 4;
				if (((j & GPUBitBoard::L5Mask) << 5) & empty)
					GenMoreBlackJumps(j << 5, out, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j << 5), (board.kings & (~cell ^ j)) | ((((board.kings & cell) << 4) << 5) | ((j << 5) & GPUBitBoard::BlackKingMask))));
			}
			//LL
			if ((((cell & board.kings) & GPUBitBoard::R5Mask) >> 5) & board.white)
			{
				GPUBitBoard::board_type j = cell >> 5;
				if ((j >> 4) & empty)
					GenMoreBlackJumps(j >> 4, out, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j >> 4), (board.kings & (~cell ^ j)) | (j >> 4)));
			}
			//LR
			if (((cell & board.kings) >> 4) & board.white)
			{
				GPUBitBoard::board_type j = cell >> 4;
				if (((j & GPUBitBoard::R3Mask) >> 3) & empty)
					GenMoreBlackJumps(j >> 3, out, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j >> 3), (board.kings & (~cell ^ j)) | (j >> 3)));
			}
		}
	}
}