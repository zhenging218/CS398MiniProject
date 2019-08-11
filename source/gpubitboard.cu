#include "precomp.cuh"

namespace Checkers
{
	__host__ __device__ GPUBitBoard::GPUBitBoard() : white(0xFFF00000u), black(0x00000FFFu), kings(0u)
	{

	}

	__host__ __device__ GPUBitBoard::GPUBitBoard(board_type w, board_type b, board_type k) : white(w), black(b), kings(k)
	{

	}

	__host__ __device__ GPUBitBoard &GPUBitBoard::operator=(GPUBitBoard const &src)
	{
		white = src.white;
		black = src.black;
		kings = src.kings;
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

	__host__ __device__ void GPUBitBoard::GenMoreWhiteJumps(GPUBitBoard::board_type cell, GPUBitBoard const &board, GPUBitBoard *frontier, int & frontier_size)
	{
		GPUBitBoard j_f[32];
		GPUBitBoard::board_type c_f[32];
		int f_size = 1;

		while (f_size > 0)
		{
			GPUBitBoard curr_j = j_f[f_size - 1];
			GPUBitBoard::board_type curr_c = c_f[f_size - 1];
			--f_size;
			GPUBitBoard::board_type jumps = GPUBitBoard::GetWhiteJumps(curr_j);
			GPUBitBoard::board_type empty = ~(curr_j.white | curr_j.black);

			if (jumps && (jumps & curr_c))
			{
				if (OddRows & curr_c)
				{
					//UL
					if (((curr_c & curr_j.kings) << 4) & curr_j.black)
					{
						GPUBitBoard::board_type j = curr_c << 4;
						if (((j & GPUBitBoard::L3Mask) << 3) & empty)
						{
							c_f[f_size] = j << 3;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~curr_c) | (j << 3), (curr_j.black & ~j), (curr_j.kings & (~curr_c ^ j)) | (j << 3));
							++f_size;
						}
					}
					//UR
					if ((((curr_c & curr_j.kings) & GPUBitBoard::L5Mask) << 5) & curr_j.black)
					{
						GPUBitBoard::board_type j = curr_c << 5;
						if ((j << 4) & empty)
						{
							c_f[f_size] = j << 4;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~curr_c) | (j << 4), (curr_j.black & ~j), (curr_j.kings & (~curr_c ^ j)) | (j << 4));
							++f_size;
						}
					}
					//LL
					if ((curr_c >> 4) & curr_j.black)
					{
						GPUBitBoard::board_type j = curr_c >> 4;
						if (((j & GPUBitBoard::R5Mask) >> 5) & empty)
						{
							c_f[f_size] = j >> 5;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~curr_c) | (j >> 5), (curr_j.black & ~j), (curr_j.kings & (~curr_c ^ j)) | ((((curr_j.kings & curr_c) >> 4) >> 5) | ((j >> 5) & GPUBitBoard::WhiteKingMask)));
							++f_size;
						}
					}
					//LR
					if (((curr_c & GPUBitBoard::R3Mask) >> 3)& curr_j.black)
					{
						GPUBitBoard::board_type j = curr_c >> 3;
						if ((j >> 4) & empty)
						{
							c_f[f_size] = j >> 4;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~curr_c) | (j >> 4), (curr_j.black & ~j), (curr_j.kings & (~curr_c ^ j)) | ((((curr_j.kings & curr_c) >> 3) >> 4) | ((j >> 4) & GPUBitBoard::WhiteKingMask)));
							++f_size;
						}
					}
				}
				else
				{
					//UL
					if ((((curr_c & GPUBitBoard::L3Mask) & curr_j.kings) << 3) & curr_j.black)
					{

						GPUBitBoard::board_type j = curr_c << 3;
						if ((j << 4) & empty)
						{
							c_f[f_size] = j << 4;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~curr_c) | (j << 4), (curr_j.black & ~j), (curr_j.kings & (~curr_c ^ j)) | (j << 4));
							++f_size;
						}
					}
					//UR
					if (((curr_c & curr_j.kings) << 4) & curr_j.black)
					{
						GPUBitBoard::board_type j = curr_c << 4;
						if (((j & GPUBitBoard::L5Mask) << 5) & empty)
						{
							c_f[f_size] = j << 5;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~curr_c) | (j << 5), (curr_j.black & ~j), (curr_j.kings & (~curr_c ^ j)) | (j << 5));
							++f_size;
						}
					}
					//LL
					if (((curr_c & GPUBitBoard::R5Mask) >> 5) & curr_j.black)
					{
						GPUBitBoard::board_type j = curr_c >> 5;
						if ((j >> 4) & empty)
						{
							c_f[f_size] = j >> 4;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~curr_c) | (j >> 4), (curr_j.black & ~j), (curr_j.kings & (~curr_c ^ j)) | ((((curr_j.kings & curr_c) >> 5) >> 4) | ((j >> 4) & GPUBitBoard::WhiteKingMask)));
							++f_size;
						}
					}
					//LR
					if ((curr_c >> 4) & curr_j.black)
					{
						GPUBitBoard::board_type j = curr_c >> 4;
						if (((j & GPUBitBoard::R3Mask) >> 3) & empty)
						{
							c_f[f_size] = j >> 3;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~curr_c) | (j >> 3), (curr_j.black & ~j), (curr_j.kings & (~curr_c ^ j)) | ((((curr_j.kings & curr_c) >> 4) >> 3) | ((j >> 3) & GPUBitBoard::WhiteKingMask)));
							++f_size;
						}
					}
				}
			}
			else
			{
				frontier[frontier_size++] = curr_j;
			}
		}
	}

	__device__ void GPUBitBoard::GenMoreWhiteJumpsAtomic(GPUBitBoard::board_type cell, GPUBitBoard const &board, GPUBitBoard *frontier, int * frontier_size)
	{
		GPUBitBoard j_f[32];
		GPUBitBoard::board_type c_f[32];
		int f_size = 1;

		while (f_size > 0)
		{
			GPUBitBoard curr_j = j_f[f_size - 1];
			GPUBitBoard::board_type curr_c = c_f[f_size - 1];
			--f_size;
			GPUBitBoard::board_type jumps = GPUBitBoard::GetWhiteJumps(curr_j);
			GPUBitBoard::board_type empty = ~(curr_j.white | curr_j.black);

			if (jumps && (jumps & curr_c))
			{
				if (OddRows & curr_c)
				{
					//UL
					if (((curr_c & curr_j.kings) << 4) & curr_j.black)
					{
						GPUBitBoard::board_type j = curr_c << 4;
						if (((j & GPUBitBoard::L3Mask) << 3) & empty)
						{
							c_f[f_size] = j << 3;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~curr_c) | (j << 3), (curr_j.black & ~j), (curr_j.kings & (~curr_c ^ j)) | (j << 3));
							++f_size;
						}
					}
					//UR
					if ((((curr_c & curr_j.kings) & GPUBitBoard::L5Mask) << 5) & curr_j.black)
					{
						GPUBitBoard::board_type j = curr_c << 5;
						if ((j << 4) & empty)
						{
							c_f[f_size] = j << 4;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~curr_c) | (j << 4), (curr_j.black & ~j), (curr_j.kings & (~curr_c ^ j)) | (j << 4));
							++f_size;
						}
					}
					//LL
					if ((curr_c >> 4) & curr_j.black)
					{
						GPUBitBoard::board_type j = curr_c >> 4;
						if (((j & GPUBitBoard::R5Mask) >> 5) & empty)
						{
							c_f[f_size] = j >> 5;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~curr_c) | (j >> 5), (curr_j.black & ~j), (curr_j.kings & (~curr_c ^ j)) | ((((curr_j.kings & curr_c) >> 4) >> 5) | ((j >> 5) & GPUBitBoard::WhiteKingMask)));
							++f_size;
						}
					}
					//LR
					if (((curr_c & GPUBitBoard::R3Mask) >> 3)& curr_j.black)
					{
						GPUBitBoard::board_type j = curr_c >> 3;
						if ((j >> 4) & empty)
						{
							c_f[f_size] = j >> 4;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~curr_c) | (j >> 4), (curr_j.black & ~j), (curr_j.kings & (~curr_c ^ j)) | ((((curr_j.kings & curr_c) >> 3) >> 4) | ((j >> 4) & GPUBitBoard::WhiteKingMask)));
							++f_size;
						}
					}
				}
				else
				{
					//UL
					if ((((curr_c & GPUBitBoard::L3Mask) & curr_j.kings) << 3) & curr_j.black)
					{

						GPUBitBoard::board_type j = curr_c << 3;
						if ((j << 4) & empty)
						{
							c_f[f_size] = j << 4;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~curr_c) | (j << 4), (curr_j.black & ~j), (curr_j.kings & (~curr_c ^ j)) | (j << 4));
							++f_size;
						}
					}
					//UR
					if (((curr_c & curr_j.kings) << 4) & curr_j.black)
					{
						GPUBitBoard::board_type j = curr_c << 4;
						if (((j & GPUBitBoard::L5Mask) << 5) & empty)
						{
							c_f[f_size] = j << 5;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~curr_c) | (j << 5), (curr_j.black & ~j), (curr_j.kings & (~curr_c ^ j)) | (j << 5));
							++f_size;
						}
					}
					//LL
					if (((curr_c & GPUBitBoard::R5Mask) >> 5) & curr_j.black)
					{
						GPUBitBoard::board_type j = curr_c >> 5;
						if ((j >> 4) & empty)
						{
							c_f[f_size] = j >> 4;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~curr_c) | (j >> 4), (curr_j.black & ~j), (curr_j.kings & (~curr_c ^ j)) | ((((curr_j.kings & curr_c) >> 5) >> 4) | ((j >> 4) & GPUBitBoard::WhiteKingMask)));
							++f_size;
						}
					}
					//LR
					if ((curr_c >> 4) & curr_j.black)
					{
						GPUBitBoard::board_type j = curr_c >> 4;
						if (((j & GPUBitBoard::R3Mask) >> 3) & empty)
						{
							c_f[f_size] = j >> 3;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~curr_c) | (j >> 3), (curr_j.black & ~j), (curr_j.kings & (~curr_c ^ j)) | ((((curr_j.kings & curr_c) >> 4) >> 3) | ((j >> 3) & GPUBitBoard::WhiteKingMask)));
							++f_size;
						}
					}
				}
			}
			else
			{
				frontier[atomicAdd(frontier_size, 1)] = curr_j;
			}
		}
	}

	__host__ __device__ void GPUBitBoard::GenMoreBlackJumps(GPUBitBoard::board_type cell, GPUBitBoard const & board, GPUBitBoard *frontier, int & frontier_size)
	{
		GPUBitBoard j_f[32];
		GPUBitBoard::board_type c_f[32];
		int f_size = 1;

		j_f[0] = board;
		c_f[0] = cell;

		while (f_size > 0)
		{
			GPUBitBoard curr_j = j_f[f_size - 1];
			GPUBitBoard::board_type curr_c = c_f[f_size - 1];
			--f_size;
			GPUBitBoard::board_type jumps = GPUBitBoard::GetBlackJumps(curr_j);
			GPUBitBoard::board_type empty = ~(curr_j.white | curr_j.black);

			if (jumps && (jumps & curr_c))
			{
				if (OddRows & curr_c)
				{
					//UL
					if ((curr_c << 4) & curr_j.white)
					{
						GPUBitBoard::board_type j = curr_c << 4;
						if (((j & GPUBitBoard::L3Mask) << 3) & empty)
						{
							c_f[f_size] = j << 3;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~j), (curr_j.black & ~curr_c) | (j << 3), (curr_j.kings & (~curr_c ^ j)) | ((((curr_j.kings & curr_c) << 4) << 3) | ((j << 3) & GPUBitBoard::BlackKingMask)));
							++f_size;
						}
					}
					//UR
					if (((curr_c & GPUBitBoard::L5Mask) << 5) & curr_j.white)
					{
						GPUBitBoard::board_type j = curr_c << 5;
						if ((j << 4) & empty)
						{
							c_f[f_size] = j << 4;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~j), (curr_j.black & ~curr_c) | (j << 4), (curr_j.kings & (~curr_c ^ j)) | ((((curr_j.kings & curr_c) << 5) << 4) | ((j << 4) & GPUBitBoard::BlackKingMask)));
							++f_size;
						}
					}
					//LL
					if (((curr_c & curr_j.kings) >> 4) & curr_j.white)
					{
						GPUBitBoard::board_type j = curr_c >> 4;
						if (((j & GPUBitBoard::R5Mask) >> 5) & empty)
						{
							c_f[f_size] = j >> 5;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~j), (curr_j.black & ~curr_c) | (j >> 5), (curr_j.kings & (~curr_c ^ j)) | (j >> 5));
							++f_size;
						}
					}
					//LR
					if ((((curr_c & GPUBitBoard::R3Mask) & curr_j.kings) >> 3) & curr_j.white)
					{
						GPUBitBoard::board_type j = curr_c >> 3;
						if ((j >> 4) & empty)
						{
							c_f[f_size] = j >> 4;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~j), (curr_j.black & ~curr_c) | (j >> 4), (curr_j.kings & (~curr_c ^ j)) | (j >> 4));
							++f_size;
						}
					}
				}
				else
				{
					//UL
					if (((curr_c & GPUBitBoard::L3Mask) << 3) & curr_j.white)
					{
						GPUBitBoard::board_type j = curr_c << 3;
						if ((j << 4) & empty)
						{
							c_f[f_size] = j << 4;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~j), (curr_j.black & ~curr_c) | (j << 4), (curr_j.kings & (~curr_c ^ j)) | ((((curr_j.kings & curr_c) << 3) << 4) | ((j << 4) & GPUBitBoard::BlackKingMask)));
							++f_size;
						}
					}
					//UR
					if ((curr_c << 4) & curr_j.white)
					{
						GPUBitBoard::board_type j = curr_c << 4;
						if (((j & GPUBitBoard::L5Mask) << 5) & empty)
						{
							c_f[f_size] = j << 5;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~j), (curr_j.black & ~curr_c) | (j << 5), (curr_j.kings & (~curr_c ^ j)) | ((((curr_j.kings & curr_c) << 4) << 5) | ((j << 5) & GPUBitBoard::BlackKingMask)));
							++f_size;
						}
					}
					//LL
					if ((((curr_c & curr_j.kings) & GPUBitBoard::R5Mask) >> 5) & curr_j.white)
					{
						GPUBitBoard::board_type j = curr_c >> 5;
						if ((j >> 4) & empty)
						{
							c_f[f_size] = j >> 4;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~j), (curr_j.black & ~curr_c) | (j >> 4), (curr_j.kings & (~curr_c ^ j)) | (j >> 4));
							++f_size;
						}
					}
					//LR
					if (((curr_c & curr_j.kings) >> 4) & curr_j.white)
					{
						GPUBitBoard::board_type j = curr_c >> 4;
						if (((j & GPUBitBoard::R3Mask) >> 3) & empty)
						{
							c_f[f_size] = j >> 3;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~j), (curr_j.black & ~curr_c) | (j >> 3), (curr_j.kings & (~curr_c ^ j)) | (j >> 3));
							++f_size;
						}
					}
				}
			}
			else
			{
				frontier[frontier_size++] = curr_j;
			}
		}
	}

	__device__ void GPUBitBoard::GenMoreBlackJumpsAtomic(GPUBitBoard::board_type cell, GPUBitBoard const & board, GPUBitBoard *frontier, int * frontier_size)
	{
		GPUBitBoard j_f[32];
		GPUBitBoard::board_type c_f[32];
		int f_size = 1;

		j_f[0] = board;
		c_f[0] = cell;

		while (f_size > 0)
		{
			GPUBitBoard curr_j = j_f[f_size - 1];
			GPUBitBoard::board_type curr_c = c_f[f_size - 1];
			--f_size;
			GPUBitBoard::board_type jumps = GPUBitBoard::GetBlackJumps(curr_j);
			GPUBitBoard::board_type empty = ~(curr_j.white | curr_j.black);

			if (jumps && (jumps & curr_c))
			{
				if (OddRows & curr_c)
				{
					//UL
					if ((curr_c << 4) & curr_j.white)
					{
						GPUBitBoard::board_type j = curr_c << 4;
						if (((j & GPUBitBoard::L3Mask) << 3) & empty)
						{
							c_f[f_size] = j << 3;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~j), (curr_j.black & ~curr_c) | (j << 3), (curr_j.kings & (~curr_c ^ j)) | ((((curr_j.kings & curr_c) << 4) << 3) | ((j << 3) & GPUBitBoard::BlackKingMask)));
							++f_size;
						}
					}
					//UR
					if (((curr_c & GPUBitBoard::L5Mask) << 5) & curr_j.white)
					{
						GPUBitBoard::board_type j = curr_c << 5;
						if ((j << 4) & empty)
						{
							c_f[f_size] = j << 4;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~j), (curr_j.black & ~curr_c) | (j << 4), (curr_j.kings & (~curr_c ^ j)) | ((((curr_j.kings & curr_c) << 5) << 4) | ((j << 4) & GPUBitBoard::BlackKingMask)));
							++f_size;
						}
					}
					//LL
					if (((curr_c & curr_j.kings) >> 4) & curr_j.white)
					{
						GPUBitBoard::board_type j = curr_c >> 4;
						if (((j & GPUBitBoard::R5Mask) >> 5) & empty)
						{
							c_f[f_size] = j >> 5;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~j), (curr_j.black & ~curr_c) | (j >> 5), (curr_j.kings & (~curr_c ^ j)) | (j >> 5));
							++f_size;
						}
					}
					//LR
					if ((((curr_c & GPUBitBoard::R3Mask) & curr_j.kings) >> 3) & curr_j.white)
					{
						GPUBitBoard::board_type j = curr_c >> 3;
						if ((j >> 4) & empty)
						{
							c_f[f_size] = j >> 4;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~j), (curr_j.black & ~curr_c) | (j >> 4), (curr_j.kings & (~curr_c ^ j)) | (j >> 4));
							++f_size;
						}
					}
				}
				else
				{
					//UL
					if (((curr_c & GPUBitBoard::L3Mask) << 3) & curr_j.white)
					{
						GPUBitBoard::board_type j = curr_c << 3;
						if ((j << 4) & empty)
						{
							c_f[f_size] = j << 4;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~j), (curr_j.black & ~curr_c) | (j << 4), (curr_j.kings & (~curr_c ^ j)) | ((((curr_j.kings & curr_c) << 3) << 4) | ((j << 4) & GPUBitBoard::BlackKingMask)));
							++f_size;
						}
					}
					//UR
					if ((curr_c << 4) & curr_j.white)
					{
						GPUBitBoard::board_type j = curr_c << 4;
						if (((j & GPUBitBoard::L5Mask) << 5) & empty)
						{
							c_f[f_size] = j << 5;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~j), (curr_j.black & ~curr_c) | (j << 5), (curr_j.kings & (~curr_c ^ j)) | ((((curr_j.kings & curr_c) << 4) << 5) | ((j << 5) & GPUBitBoard::BlackKingMask)));
							++f_size;
						}
					}
					//LL
					if ((((curr_c & curr_j.kings) & GPUBitBoard::R5Mask) >> 5) & curr_j.white)
					{
						GPUBitBoard::board_type j = curr_c >> 5;
						if ((j >> 4) & empty)
						{
							c_f[f_size] = j >> 4;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~j), (curr_j.black & ~curr_c) | (j >> 4), (curr_j.kings & (~curr_c ^ j)) | (j >> 4));
							++f_size;
						}
					}
					//LR
					if (((curr_c & curr_j.kings) >> 4) & curr_j.white)
					{
						GPUBitBoard::board_type j = curr_c >> 4;
						if (((j & GPUBitBoard::R3Mask) >> 3) & empty)
						{
							c_f[f_size] = j >> 3;
							j_f[f_size] = GPUBitBoard((curr_j.white & ~j), (curr_j.black & ~curr_c) | (j >> 3), (curr_j.kings & (~curr_c ^ j)) | (j >> 3));
							++f_size;
						}
					}
				}
			}
			else
			{
				frontier[atomicAdd(frontier_size, 1)] = curr_j;
			}
		}
	}

	__host__ __device__ void GPUBitBoard::GenWhiteMove(GPUBitBoard::board_type cell, GPUBitBoard const &board, GPUBitBoard *frontier, int & frontier_size)
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
				frontier[frontier_size++] = GPUBitBoard((board.white & ~cell) | (cell << 4), board.black, (board.kings & ~cell) | (cell << 4));
			}
			//UR
			if ((((cell & board.kings) & GPUBitBoard::L5Mask) << 5) & empty)
			{
				frontier[frontier_size++] = GPUBitBoard((board.white & ~cell) | (cell << 5), board.black, (board.kings & ~cell) | (cell << 5));
			}
			//LL
			if ((cell >> 4) & empty)
			{
				frontier[frontier_size++] = GPUBitBoard((board.white & ~cell) | (cell >> 4), board.black, (board.kings & ~cell) | (((board.kings & cell) >> 4) | ((cell >> 4) & GPUBitBoard::WhiteKingMask)));
			}
			//LR
			if (((cell & GPUBitBoard::R3Mask) >> 3)& empty)
			{
				frontier[frontier_size++] = GPUBitBoard((board.white & ~cell) | (cell >> 3), board.black, (board.kings & ~cell) | (((board.kings & cell) >> 3) | ((cell >> 3) & GPUBitBoard::WhiteKingMask)));
			}
		}
		else
		{
			//UL
			if ((((cell & GPUBitBoard::L3Mask) & board.kings) << 3) & empty)
			{
				frontier[frontier_size++] = GPUBitBoard((board.white & ~cell) | (cell << 3), board.black, (board.kings & ~cell) | (cell << 3));
			}
			//UR
			if (((cell & board.kings) << 4) & empty)
			{
				frontier[frontier_size++] = GPUBitBoard((board.white & ~cell) | (cell << 4), board.black, (board.kings & ~cell) | (cell << 4));
			}
			//LL
			if (((cell & GPUBitBoard::R5Mask) >> 5) & empty)
			{
				frontier[frontier_size++] = GPUBitBoard((board.white & ~cell) | (cell >> 5), board.black, (board.kings & ~cell) | (((board.kings & cell) >> 5) | ((cell >> 5) & GPUBitBoard::WhiteKingMask)));
			}
			//LR
			if ((cell >> 4) & empty)
			{
				frontier[frontier_size++] = GPUBitBoard((board.white & ~cell) | (cell >> 4), board.black, (board.kings & ~cell) | (((board.kings & cell) >> 4) | ((cell >> 4) & GPUBitBoard::WhiteKingMask)));
			}
		}
	}

	__device__ void GPUBitBoard::GenWhiteMoveAtomic(GPUBitBoard::board_type cell, GPUBitBoard const &board, GPUBitBoard *frontier, int * frontier_size)
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
				frontier[atomicAdd(frontier_size, 1)] = GPUBitBoard((board.white & ~cell) | (cell << 4), board.black, (board.kings & ~cell) | (cell << 4));
			}
			//UR
			if ((((cell & board.kings) & GPUBitBoard::L5Mask) << 5) & empty)
			{
				frontier[atomicAdd(frontier_size, 1)] = GPUBitBoard((board.white & ~cell) | (cell << 5), board.black, (board.kings & ~cell) | (cell << 5));
			}
			//LL
			if ((cell >> 4) & empty)
			{
				frontier[atomicAdd(frontier_size, 1)] = GPUBitBoard((board.white & ~cell) | (cell >> 4), board.black, (board.kings & ~cell) | (((board.kings & cell) >> 4) | ((cell >> 4) & GPUBitBoard::WhiteKingMask)));
			}
			//LR
			if (((cell & GPUBitBoard::R3Mask) >> 3)& empty)
			{
				frontier[atomicAdd(frontier_size, 1)] = GPUBitBoard((board.white & ~cell) | (cell >> 3), board.black, (board.kings & ~cell) | (((board.kings & cell) >> 3) | ((cell >> 3) & GPUBitBoard::WhiteKingMask)));
			}
		}
		else
		{
			//UL
			if ((((cell & GPUBitBoard::L3Mask) & board.kings) << 3) & empty)
			{
				frontier[atomicAdd(frontier_size, 1)] = GPUBitBoard((board.white & ~cell) | (cell << 3), board.black, (board.kings & ~cell) | (cell << 3));
			}
			//UR
			if (((cell & board.kings) << 4) & empty)
			{
				frontier[atomicAdd(frontier_size, 1)] = GPUBitBoard((board.white & ~cell) | (cell << 4), board.black, (board.kings & ~cell) | (cell << 4));
			}
			//LL
			if (((cell & GPUBitBoard::R5Mask) >> 5) & empty)
			{
				frontier[atomicAdd(frontier_size, 1)] = GPUBitBoard((board.white & ~cell) | (cell >> 5), board.black, (board.kings & ~cell) | (((board.kings & cell) >> 5) | ((cell >> 5) & GPUBitBoard::WhiteKingMask)));
			}
			//LR
			if ((cell >> 4) & empty)
			{
				frontier[atomicAdd(frontier_size, 1)] = GPUBitBoard((board.white & ~cell) | (cell >> 4), board.black, (board.kings & ~cell) | (((board.kings & cell) >> 4) | ((cell >> 4) & GPUBitBoard::WhiteKingMask)));
			}
		}
	}

	__host__ __device__ void GPUBitBoard::GenWhiteJump(GPUBitBoard::board_type cell, GPUBitBoard const &board, GPUBitBoard *frontier, int & frontier_size)
	{
		GPUBitBoard::board_type empty = ~(board.white | board.black);
		if (OddRows & cell)
		{
			//UL
			if (((cell & board.kings) << 4) & board.black)
			{
				GPUBitBoard::board_type j = cell << 4;
				if (((j & GPUBitBoard::L3Mask) << 3) & empty)
					GenMoreWhiteJumps(j << 3, GPUBitBoard((board.white & ~cell) | (j << 3), (board.black & ~j), (board.kings & (~cell ^ j)) | (j << 3)), frontier, frontier_size);
			}
			//UR
			if ((((cell & board.kings) & GPUBitBoard::L5Mask) << 5) & board.black)
			{
				GPUBitBoard::board_type j = cell << 5;
				if ((j << 4) & empty)
					GenMoreWhiteJumps(j << 4,  GPUBitBoard((board.white & ~cell) | (j << 4), (board.black & ~j), (board.kings & (~cell ^ j)) | (j << 4)), frontier, frontier_size);
			}
			//LL
			if ((cell >> 4) & board.black)
			{
				GPUBitBoard::board_type j = cell >> 4;
				if (((j & GPUBitBoard::R5Mask) >> 5) & empty)
					GenMoreWhiteJumps(j >> 5, GPUBitBoard((board.white & ~cell) | (j >> 5), (board.black & ~j), (board.kings & (~cell ^ j)) | ((((board.kings & cell) >> 4) >> 5) | ((j >> 5) & GPUBitBoard::WhiteKingMask))),frontier, frontier_size);
			}
			//LR
			if (((cell & GPUBitBoard::R3Mask) >> 3)& board.black)
			{
				GPUBitBoard::board_type j = cell >> 3;
				if ((j >> 4) & empty)
					GenMoreWhiteJumps(j >> 4, GPUBitBoard((board.white & ~cell) | (j >> 4), (board.black & ~j), (board.kings & (~cell ^ j)) | ((((board.kings & cell) >> 3) >> 4) | ((j >> 4) & GPUBitBoard::WhiteKingMask))),frontier, frontier_size);
			}
		}
		else
		{
			//UL
			if ((((cell & GPUBitBoard::L3Mask) & board.kings) << 3) & board.black)
			{

				GPUBitBoard::board_type j = cell << 3;
				if ((j << 4) & empty)
					GenMoreWhiteJumps(j << 4, GPUBitBoard((board.white & ~cell) | (j << 4), (board.black & ~j), (board.kings & (~cell ^ j)) | (j << 4)),frontier, frontier_size);
			}
			//UR
			if (((cell & board.kings) << 4) & board.black)
			{
				GPUBitBoard::board_type j = cell << 4;
				if (((j & GPUBitBoard::L5Mask) << 5) & empty)
					GenMoreWhiteJumps(j << 5, GPUBitBoard((board.white & ~cell) | (j << 5), (board.black & ~j), (board.kings & (~cell ^ j)) | (j << 5)),frontier, frontier_size);
			}
			//LL
			if (((cell & GPUBitBoard::R5Mask) >> 5) & board.black)
			{
				GPUBitBoard::board_type j = cell >> 5;
				if ((j >> 4) & empty)
					GenMoreWhiteJumps(j >> 4, GPUBitBoard((board.white & ~cell) | (j >> 4), (board.black & ~j), (board.kings & (~cell ^ j)) | ((((board.kings & cell) >> 5) >> 4) | ((j >> 4) & GPUBitBoard::WhiteKingMask))),frontier, frontier_size);
			}
			//LR
			if ((cell >> 4) & board.black)
			{
				GPUBitBoard::board_type j = cell >> 4;
				if (((j & GPUBitBoard::R3Mask) >> 3) & empty)
					GenMoreWhiteJumps(j >> 3, GPUBitBoard((board.white & ~cell) | (j >> 3), (board.black & ~j), (board.kings & (~cell ^ j)) | ((((board.kings & cell) >> 4) >> 3) | ((j >> 3) & GPUBitBoard::WhiteKingMask))),frontier, frontier_size);
			}
		}
	}

	__device__ void GPUBitBoard::GenWhiteJumpAtomic(GPUBitBoard::board_type cell, GPUBitBoard const &board, GPUBitBoard *frontier, int * frontier_size)
	{
		GPUBitBoard::board_type empty = ~(board.white | board.black);
		if (OddRows & cell)
		{
			//UL
			if (((cell & board.kings) << 4) & board.black)
			{
				GPUBitBoard::board_type j = cell << 4;
				if (((j & GPUBitBoard::L3Mask) << 3) & empty)
					GenMoreWhiteJumpsAtomic(j << 3, GPUBitBoard((board.white & ~cell) | (j << 3), (board.black & ~j), (board.kings & (~cell ^ j)) | (j << 3)),frontier, frontier_size);
			}
			//UR
			if ((((cell & board.kings) & GPUBitBoard::L5Mask) << 5) & board.black)
			{
				GPUBitBoard::board_type j = cell << 5;
				if ((j << 4) & empty)
					GenMoreWhiteJumpsAtomic(j << 4, GPUBitBoard((board.white & ~cell) | (j << 4), (board.black & ~j), (board.kings & (~cell ^ j)) | (j << 4)),frontier, frontier_size);
			}
			//LL
			if ((cell >> 4) & board.black)
			{
				GPUBitBoard::board_type j = cell >> 4;
				if (((j & GPUBitBoard::R5Mask) >> 5) & empty)
					GenMoreWhiteJumpsAtomic(j >> 5, GPUBitBoard((board.white & ~cell) | (j >> 5), (board.black & ~j), (board.kings & (~cell ^ j)) | ((((board.kings & cell) >> 4) >> 5) | ((j >> 5) & GPUBitBoard::WhiteKingMask))),frontier, frontier_size);
			}
			//LR
			if (((cell & GPUBitBoard::R3Mask) >> 3)& board.black)
			{
				GPUBitBoard::board_type j = cell >> 3;
				if ((j >> 4) & empty)
					GenMoreWhiteJumpsAtomic(j >> 4, GPUBitBoard((board.white & ~cell) | (j >> 4), (board.black & ~j), (board.kings & (~cell ^ j)) | ((((board.kings & cell) >> 3) >> 4) | ((j >> 4) & GPUBitBoard::WhiteKingMask))),frontier, frontier_size);
			}
		}
		else
		{
			//UL
			if ((((cell & GPUBitBoard::L3Mask) & board.kings) << 3) & board.black)
			{

				GPUBitBoard::board_type j = cell << 3;
				if ((j << 4) & empty)
					GenMoreWhiteJumpsAtomic(j << 4, GPUBitBoard((board.white & ~cell) | (j << 4), (board.black & ~j), (board.kings & (~cell ^ j)) | (j << 4)),frontier, frontier_size);
			}
			//UR
			if (((cell & board.kings) << 4) & board.black)
			{
				GPUBitBoard::board_type j = cell << 4;
				if (((j & GPUBitBoard::L5Mask) << 5) & empty)
					GenMoreWhiteJumpsAtomic(j << 5, GPUBitBoard((board.white & ~cell) | (j << 5), (board.black & ~j), (board.kings & (~cell ^ j)) | (j << 5)),frontier, frontier_size);
			}
			//LL
			if (((cell & GPUBitBoard::R5Mask) >> 5) & board.black)
			{
				GPUBitBoard::board_type j = cell >> 5;
				if ((j >> 4) & empty)
					GenMoreWhiteJumpsAtomic(j >> 4, GPUBitBoard((board.white & ~cell) | (j >> 4), (board.black & ~j), (board.kings & (~cell ^ j)) | ((((board.kings & cell) >> 5) >> 4) | ((j >> 4) & GPUBitBoard::WhiteKingMask))),frontier, frontier_size);
			}
			//LR
			if ((cell >> 4) & board.black)
			{
				GPUBitBoard::board_type j = cell >> 4;
				if (((j & GPUBitBoard::R3Mask) >> 3) & empty)
					GenMoreWhiteJumpsAtomic(j >> 3, GPUBitBoard((board.white & ~cell) | (j >> 3), (board.black & ~j), (board.kings & (~cell ^ j)) | ((((board.kings & cell) >> 4) >> 3) | ((j >> 3) & GPUBitBoard::WhiteKingMask))),frontier, frontier_size);
			}
		}
	}

	__host__ __device__ void GPUBitBoard::GenBlackMove(GPUBitBoard::board_type cell, GPUBitBoard const &board, GPUBitBoard *frontier, int & frontier_size)
	{
		// GPUBitBoard must have a validity boolean.

		GPUBitBoard::board_type empty = ~(board.white | board.black);
		if (OddRows & cell)
		{
			//UL
			if ((cell << 4) & empty)
			{
				frontier[frontier_size++] = GPUBitBoard(board.white, (board.black & ~cell) | (cell << 4), (board.kings & ~cell) | (((board.kings & cell) << 4) | ((cell << 4) & GPUBitBoard::BlackKingMask)));
			}
			//UR
			if (((cell & GPUBitBoard::L5Mask) << 5) & empty)
			{
				frontier[frontier_size++] = GPUBitBoard(board.white, (board.black & ~cell) | (cell << 5), (board.kings & ~cell) | (((board.kings & cell) << 5) | ((cell << 5) & GPUBitBoard::BlackKingMask)));
			}
			//LL
			if (((cell & board.kings) >> 4) & empty)
			{
				frontier[frontier_size++] = GPUBitBoard(board.white, (board.black & ~cell) | (cell >> 4), (board.kings & ~cell) | (cell >> 4));
			}
			//LR
			if ((((cell & GPUBitBoard::R3Mask)& board.kings) >> 3)& empty)
			{
				frontier[frontier_size++] = GPUBitBoard(board.white, (board.black & ~cell) | (cell >> 3), (board.kings & ~cell) | (cell >> 3));
			}
		}
		else
		{
			//UL
			if (((cell & GPUBitBoard::L3Mask) << 3) & empty)
			{
				frontier[frontier_size++] = GPUBitBoard(board.white, (board.black & ~cell) | (cell << 3), (board.kings & ~cell) | (((board.kings & cell) << 3) | ((cell << 3) & GPUBitBoard::BlackKingMask)));
			}
			//UR
			if ((cell << 4) & empty)
			{
				frontier[frontier_size++] = GPUBitBoard(board.white, (board.black & ~cell) | (cell << 4), (board.kings & ~cell) | (((board.kings & cell) << 4) | ((cell << 4) & GPUBitBoard::BlackKingMask)));
			}
			//LL
			if ((((cell & board.kings) & GPUBitBoard::R5Mask) >> 5) & empty)
			{
				frontier[frontier_size++] = GPUBitBoard(board.white, (board.black & ~cell) | (cell >> 5), (board.kings & ~cell) | (cell >> 5));
			}
			//LR
			if (((cell & board.kings) >> 4) & empty)
			{
				frontier[frontier_size++] = GPUBitBoard(board.white, (board.black & ~cell) | (cell >> 4), (board.kings & ~cell) | (cell >> 4));
			}
		}
	}

	__device__ void GPUBitBoard::GenBlackMoveAtomic(GPUBitBoard::board_type cell, GPUBitBoard const &board, GPUBitBoard *frontier, int * frontier_size)
	{
		// GPUBitBoard must have a validity boolean.

		GPUBitBoard::board_type empty = ~(board.white | board.black);
		if (OddRows & cell)
		{
			//UL
			if ((cell << 4) & empty)
			{
				frontier[atomicAdd(frontier_size, 1)] = GPUBitBoard(board.white, (board.black & ~cell) | (cell << 4), (board.kings & ~cell) | (((board.kings & cell) << 4) | ((cell << 4) & GPUBitBoard::BlackKingMask)));
			}
			//UR
			if (((cell & GPUBitBoard::L5Mask) << 5) & empty)
			{
				frontier[atomicAdd(frontier_size, 1)] = GPUBitBoard(board.white, (board.black & ~cell) | (cell << 5), (board.kings & ~cell) | (((board.kings & cell) << 5) | ((cell << 5) & GPUBitBoard::BlackKingMask)));
			}
			//LL
			if (((cell & board.kings) >> 4) & empty)
			{
				frontier[atomicAdd(frontier_size, 1)] = GPUBitBoard(board.white, (board.black & ~cell) | (cell >> 4), (board.kings & ~cell) | (cell >> 4));
			}
			//LR
			if ((((cell & GPUBitBoard::R3Mask)& board.kings) >> 3)& empty)
			{
				frontier[atomicAdd(frontier_size, 1)] = GPUBitBoard(board.white, (board.black & ~cell) | (cell >> 3), (board.kings & ~cell) | (cell >> 3));
			}
		}
		else
		{
			//UL
			if (((cell & GPUBitBoard::L3Mask) << 3) & empty)
			{
				frontier[atomicAdd(frontier_size, 1)] = GPUBitBoard(board.white, (board.black & ~cell) | (cell << 3), (board.kings & ~cell) | (((board.kings & cell) << 3) | ((cell << 3) & GPUBitBoard::BlackKingMask)));
			}
			//UR
			if ((cell << 4) & empty)
			{
				frontier[atomicAdd(frontier_size, 1)] = GPUBitBoard(board.white, (board.black & ~cell) | (cell << 4), (board.kings & ~cell) | (((board.kings & cell) << 4) | ((cell << 4) & GPUBitBoard::BlackKingMask)));
			}
			//LL
			if ((((cell & board.kings) & GPUBitBoard::R5Mask) >> 5) & empty)
			{
				frontier[atomicAdd(frontier_size, 1)] = GPUBitBoard(board.white, (board.black & ~cell) | (cell >> 5), (board.kings & ~cell) | (cell >> 5));
			}
			//LR
			if (((cell & board.kings) >> 4) & empty)
			{
				frontier[atomicAdd(frontier_size, 1)] = GPUBitBoard(board.white, (board.black & ~cell) | (cell >> 4), (board.kings & ~cell) | (cell >> 4));
			}
		}
	}

	__host__ __device__ void GPUBitBoard::GenBlackJump(GPUBitBoard::board_type cell, GPUBitBoard const &board, GPUBitBoard *frontier, int & frontier_size)
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
					GenMoreBlackJumps(j << 3, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j << 3), (board.kings & (~cell ^ j)) | ((((board.kings & cell) << 4) << 3) | ((j << 3) & GPUBitBoard::BlackKingMask))),frontier, frontier_size);
			}
			//UR
			if (((cell & GPUBitBoard::L5Mask) << 5) & board.white)
			{
				GPUBitBoard::board_type j = cell << 5;
				if ((j << 4) & empty)
					GenMoreBlackJumps(j << 4, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j << 4), (board.kings & (~cell ^ j)) | ((((board.kings & cell) << 5) << 4) | ((j << 4) & GPUBitBoard::BlackKingMask))),frontier, frontier_size);
			}
			//LL
			if (((cell & board.kings) >> 4) & board.white)
			{
				GPUBitBoard::board_type j = cell >> 4;
				if (((j & GPUBitBoard::R5Mask) >> 5) & empty)
					GenMoreBlackJumps(j >> 5, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j >> 5), (board.kings & (~cell ^ j)) | (j >> 5)),frontier, frontier_size);
			}
			//LR
			if ((((cell & GPUBitBoard::R3Mask) & board.kings) >> 3) & board.white)
			{
				GPUBitBoard::board_type j = cell >> 3;
				if ((j >> 4) & empty)
					GenMoreBlackJumps(j >> 4, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j >> 4), (board.kings & (~cell ^ j)) | (j >> 4)),frontier, frontier_size);
			}
		}
		else
		{
			//UL
			if (((cell & GPUBitBoard::L3Mask) << 3) & board.white)
			{
				GPUBitBoard::board_type j = cell << 3;
				if ((j << 4) & empty)
					GenMoreBlackJumps(j << 4, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j << 4), (board.kings & (~cell ^ j)) | ((((board.kings & cell) << 3) << 4) | ((j << 4) & GPUBitBoard::BlackKingMask))),frontier, frontier_size);
			}
			//UR
			if ((cell << 4) & board.white)
			{
				GPUBitBoard::board_type j = cell << 4;
				if (((j & GPUBitBoard::L5Mask) << 5) & empty)
					GenMoreBlackJumps(j << 5, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j << 5), (board.kings & (~cell ^ j)) | ((((board.kings & cell) << 4) << 5) | ((j << 5) & GPUBitBoard::BlackKingMask))),frontier, frontier_size);
			}
			//LL
			if ((((cell & board.kings) & GPUBitBoard::R5Mask) >> 5) & board.white)
			{
				GPUBitBoard::board_type j = cell >> 5;
				if ((j >> 4) & empty)
					GenMoreBlackJumps(j >> 4, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j >> 4), (board.kings & (~cell ^ j)) | (j >> 4)),frontier, frontier_size);
			}
			//LR
			if (((cell & board.kings) >> 4) & board.white)
			{
				GPUBitBoard::board_type j = cell >> 4;
				if (((j & GPUBitBoard::R3Mask) >> 3) & empty)
					GenMoreBlackJumps(j >> 3, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j >> 3), (board.kings & (~cell ^ j)) | (j >> 3)),frontier, frontier_size);
			}
		}
	}

	__device__ void GPUBitBoard::GenBlackJumpAtomic(GPUBitBoard::board_type cell, GPUBitBoard const &board, GPUBitBoard *frontier, int * frontier_size)
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
					GenMoreBlackJumpsAtomic(j << 3, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j << 3), (board.kings & (~cell ^ j)) | ((((board.kings & cell) << 4) << 3) | ((j << 3) & GPUBitBoard::BlackKingMask))),frontier, frontier_size);
			}
			//UR
			if (((cell & GPUBitBoard::L5Mask) << 5) & board.white)
			{
				GPUBitBoard::board_type j = cell << 5;
				if ((j << 4) & empty)
					GenMoreBlackJumpsAtomic(j << 4, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j << 4), (board.kings & (~cell ^ j)) | ((((board.kings & cell) << 5) << 4) | ((j << 4) & GPUBitBoard::BlackKingMask))),frontier, frontier_size);
			}
			//LL
			if (((cell & board.kings) >> 4) & board.white)
			{
				GPUBitBoard::board_type j = cell >> 4;
				if (((j & GPUBitBoard::R5Mask) >> 5) & empty)
					GenMoreBlackJumpsAtomic(j >> 5, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j >> 5), (board.kings & (~cell ^ j)) | (j >> 5)),frontier, frontier_size);
			}
			//LR
			if ((((cell & GPUBitBoard::R3Mask) & board.kings) >> 3) & board.white)
			{
				GPUBitBoard::board_type j = cell >> 3;
				if ((j >> 4) & empty)
					GenMoreBlackJumpsAtomic(j >> 4, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j >> 4), (board.kings & (~cell ^ j)) | (j >> 4)),frontier, frontier_size);
			}
		}
		else
		{
			//UL
			if (((cell & GPUBitBoard::L3Mask) << 3) & board.white)
			{
				GPUBitBoard::board_type j = cell << 3;
				if ((j << 4) & empty)
					GenMoreBlackJumpsAtomic(j << 4, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j << 4), (board.kings & (~cell ^ j)) | ((((board.kings & cell) << 3) << 4) | ((j << 4) & GPUBitBoard::BlackKingMask))),frontier, frontier_size);
			}
			//UR
			if ((cell << 4) & board.white)
			{
				GPUBitBoard::board_type j = cell << 4;
				if (((j & GPUBitBoard::L5Mask) << 5) & empty)
					GenMoreBlackJumpsAtomic(j << 5, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j << 5), (board.kings & (~cell ^ j)) | ((((board.kings & cell) << 4) << 5) | ((j << 5) & GPUBitBoard::BlackKingMask))),frontier, frontier_size);
			}
			//LL
			if ((((cell & board.kings) & GPUBitBoard::R5Mask) >> 5) & board.white)
			{
				GPUBitBoard::board_type j = cell >> 5;
				if ((j >> 4) & empty)
					GenMoreBlackJumpsAtomic(j >> 4, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j >> 4), (board.kings & (~cell ^ j)) | (j >> 4)),frontier, frontier_size);
			}
			//LR
			if (((cell & board.kings) >> 4) & board.white)
			{
				GPUBitBoard::board_type j = cell >> 4;
				if (((j & GPUBitBoard::R3Mask) >> 3) & empty)
					GenMoreBlackJumpsAtomic(j >> 3, GPUBitBoard((board.white & ~j), (board.black & ~cell) | (j >> 3), (board.kings & (~cell ^ j)) | (j >> 3)),frontier, frontier_size);
			}
		}
	}
}