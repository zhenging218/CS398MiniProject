#include "precomp.h"
#include <utility>
#include <array>

#define UPPER_RIGHT(x) ((x) << 3)
#define LOWER_RIGHT(x) ((x) >> 4)
#define UPPER_LEFT(x) ((x) << 4)
#define LOWER_LEFT(x) ((x) >> 3)

#define NEXT_MOVE(x) (x >> 1)

namespace Checkers
{
	namespace
	{
		constexpr BitBoard::board_type KingMask = 0xF000000F;
		constexpr BitBoard::board_type Mask = 0xFFFFFFFF;

		BitBoard::board_type InitializeWhitePieces()
		{
			return 0xFFFF0000;
		}

		BitBoard::board_type InitializeBlackPieces()
		{
			return 0x0000FFFF;
		}
	}

	BitBoard::BitBoard() : white(InitializeWhitePieces()), black(InitializeBlackPieces()), kings(0)
	{

	}

	BitBoard::board_type BitBoard::GetLLNonJumpWhiteMoves() const
	{
		board_type not_occupied = ~(white | black);
		return UPPER_RIGHT(LOWER_LEFT(white) & not_occupied);
	}

	BitBoard::board_type BitBoard::GetLRNonJumpWhiteMoves() const
	{
		board_type not_occupied = ~(white | black);
		return UPPER_LEFT(LOWER_RIGHT(white) & not_occupied);
	}

	BitBoard::board_type BitBoard::GetURNonJumpWhiteMmoves() const
	{
		board_type not_occupied = ~(white | black);
		return LOWER_LEFT(UPPER_RIGHT(white & kings) & not_occupied);
	}

	BitBoard::board_type BitBoard::GetULNonJumpWhiteMoves() const
	{
		board_type not_occupied = ~(white | black);
		return LOWER_RIGHT(UPPER_LEFT(white & kings) & not_occupied);
	}

	BitBoard::board_type BitBoard::GetLLJumpWhiteMoves() const
	{
		board_type not_occupied = ~(white | black);
		return UPPER_RIGHT(UPPER_RIGHT(LOWER_LEFT(LOWER_LEFT(white) & black) & not_occupied));
	}

	BitBoard::board_type BitBoard::GetLRJumpWhiteMoves() const
	{
		board_type not_occupied = ~(white | black);
		return UPPER_LEFT(UPPER_LEFT(LOWER_RIGHT(LOWER_RIGHT(white) & black) &not_occupied));
	}

	BitBoard::board_type BitBoard::GetURJumpWhiteMmoves() const
	{
		board_type not_occupied = ~(white | black);
		return LOWER_LEFT(LOWER_LEFT(UPPER_RIGHT(UPPER_RIGHT(white & kings) & black) &not_occupied));
	}

	BitBoard::board_type BitBoard::GetULJumpWhiteMoves() const
	{
		board_type not_occupied = ~(white | black);
		return LOWER_RIGHT(LOWER_RIGHT(UPPER_LEFT(UPPER_LEFT(white & kings) & black) & not_occupied));
	}

	BitBoard::board_type BitBoard::GetULNonJumpBlackMoves() const
	{
		board_type not_occupied = ~(white | black);
		return LOWER_RIGHT(UPPER_LEFT(black) & not_occupied);
	}

	BitBoard::board_type BitBoard::GetURNonJumpBlackMoves() const
	{
		board_type not_occupied = ~(white | black);
		return LOWER_LEFT(UPPER_RIGHT(black) & not_occupied);
	}

	BitBoard::board_type BitBoard::GetLRNonJumpBlackMoves() const
	{
		board_type not_occupied = ~(white | black);
		return UPPER_LEFT(LOWER_RIGHT(black & kings) & not_occupied);
	}

	BitBoard::board_type BitBoard::GetLLNonJumpBlackMoves() const
	{
		board_type not_occupied = ~(white | black);
		return UPPER_RIGHT(LOWER_LEFT(black & kings) & not_occupied);
	}

	BitBoard::board_type BitBoard::GetULJumpBlackMoves() const
	{
		board_type not_occupied = ~(white | black);
		return LOWER_RIGHT(LOWER_RIGHT(UPPER_LEFT(UPPER_LEFT(black) & white) & not_occupied));
	}

	BitBoard::board_type BitBoard::GetURJumpBlackMoves() const
	{
		board_type not_occupied = ~(white | black);
		return LOWER_LEFT(LOWER_LEFT(UPPER_RIGHT(UPPER_RIGHT(black) & white) &not_occupied));
	}

	BitBoard::board_type BitBoard::GetLRJumpBlackMoves() const
	{
		board_type not_occupied = ~(white | black);
		return UPPER_LEFT(UPPER_LEFT(LOWER_RIGHT(LOWER_RIGHT(black & kings) & white) &not_occupied));
	}

	BitBoard::board_type BitBoard::GetLLJumpBlackMoves() const
	{
		board_type not_occupied = ~(white | black);
		return UPPER_RIGHT(UPPER_RIGHT(LOWER_LEFT(LOWER_LEFT(black & kings) & white) & not_occupied));
	}

	BitBoard::board_type BitBoard::GetWhiteMoves() const
	{
		board_type not_occupied = ~(white | black);
		board_type LL_non_jump_moves = UPPER_RIGHT(LOWER_LEFT(white) & not_occupied);
		board_type LR_non_jump_moves = UPPER_LEFT(LOWER_RIGHT(white) & not_occupied);
		board_type UR_non_jump_moves = LOWER_LEFT(UPPER_RIGHT(white & kings) & not_occupied);
		board_type UL_non_jump_moves = LOWER_RIGHT(UPPER_LEFT(white & kings) & not_occupied);

		return LL_non_jump_moves | LR_non_jump_moves | UR_non_jump_moves | UL_non_jump_moves;
	}

	BitBoard::board_type BitBoard::GetWhiteJumps() const
	{
		board_type not_occupied = ~(white | black);
		board_type LL_jump_moves = UPPER_RIGHT(UPPER_RIGHT(LOWER_LEFT(LOWER_LEFT(white) & black) & not_occupied));
		board_type LR_jump_moves = UPPER_LEFT(UPPER_LEFT(LOWER_RIGHT(LOWER_RIGHT(white) & black) &not_occupied));
		board_type UR_jump_moves = LOWER_LEFT(LOWER_LEFT(UPPER_RIGHT(UPPER_RIGHT(white & kings) & black) &not_occupied));
		board_type UL_jump_moves = LOWER_RIGHT(LOWER_RIGHT(UPPER_LEFT(UPPER_LEFT(white & kings) & black) & not_occupied));
		return LL_jump_moves | LR_jump_moves | UL_jump_moves | UR_jump_moves;
	}

	BitBoard::board_type BitBoard::GetBlackMoves() const
	{
		board_type not_occupied = ~(white | black);
		board_type UL_non_jump_moves = LOWER_RIGHT(UPPER_LEFT(black) & not_occupied);
		board_type UR_non_jump_moves = LOWER_LEFT(UPPER_RIGHT(black) & not_occupied);
		board_type LR_non_jump_moves = UPPER_LEFT(LOWER_RIGHT(black & kings) & not_occupied);
		board_type LL_non_jump_moves = UPPER_RIGHT(LOWER_LEFT(black & kings) & not_occupied);
		

		return LL_non_jump_moves | LR_non_jump_moves | UR_non_jump_moves | UL_non_jump_moves;
	}

	BitBoard::board_type BitBoard::GetBlackJumps() const
	{
		board_type not_occupied = ~(white | black);
		board_type UL_jump_moves = LOWER_RIGHT(LOWER_RIGHT(UPPER_LEFT(UPPER_LEFT(black) & white) & not_occupied));
		board_type UR_jump_moves = LOWER_LEFT(LOWER_LEFT(UPPER_RIGHT(UPPER_RIGHT(black) & white) &not_occupied));
		board_type LR_jump_moves = UPPER_LEFT(UPPER_LEFT(LOWER_RIGHT(LOWER_RIGHT(black & kings) & white) &not_occupied));
		board_type LL_jump_moves = UPPER_RIGHT(UPPER_RIGHT(LOWER_LEFT(LOWER_LEFT(black & kings) & white) & not_occupied));

		return LL_jump_moves | LR_jump_moves | UL_jump_moves | UR_jump_moves;
	}

	BitBoard::board_type BitBoard::GetWhiteKings() const
	{
		return kings & white;
	}

	BitBoard::board_type BitBoard::GetBlackKings() const
	{
		return kings & black;
	}

	std::vector<BitBoard> BitBoard::GetPossibleBlackMoves(BitBoard const &src)
	{
		std::vector<BitBoard> ret;

		board_type ULNonJumpMoves = src.GetULNonJumpBlackMoves();
		board_type URNonJumpMoves = src.GetURNonJumpBlackMoves();
		board_type LLNonJumpMoves = src.GetLLNonJumpBlackMoves();
		board_type LRNonJumpMoves = src.GetLRNonJumpBlackMoves();

		board_type ULJumpMoves = src.GetULJumpBlackMoves();
		board_type URJumpMoves = src.GetURJumpBlackMoves();
		board_type LLJumpMoves = src.GetLLJumpBlackMoves();
		board_type LRJumpMoves = src.GetLRJumpBlackMoves();

		for (board_type i = 1; ULNonJumpMoves && URNonJumpMoves && LLNonJumpMoves && LRNonJumpMoves; i <<= 1)
		{
			if (i & ULNonJumpMoves)
			{
				ret.push_back(BitBoard(
					src.white, 
					(src.black & (Mask ^ i)) | UPPER_LEFT(i),
					((i & src.kings) ^ src.kings) & (UPPER_LEFT(i) & KingMask)
				));
			}

			if (i & URNonJumpMoves)
			{
				ret.push_back(BitBoard(
					src.white, 
					(src.black & (Mask ^ i)) | UPPER_RIGHT(i),
					(i & src.kings) ^ src.kings & (UPPER_RIGHT(i) & KingMask)
				));
			}

			if (i & LLNonJumpMoves)
			{
				ret.push_back(BitBoard(
					src.white, 
					(src.black & (Mask ^ i)) | LOWER_LEFT(i), 
					(i & src.kings) ^ src.kings & (LOWER_LEFT(i) & KingMask)
				));
			}

			if (i & LRNonJumpMoves)
			{
				ret.push_back(BitBoard(
					src.white, 
					(src.black & (Mask ^ i)) | LOWER_RIGHT(i),
					(i & src.kings) ^ src.kings & (LOWER_RIGHT(i) & KingMask)
				));
			}

			// update the moves
			ULNonJumpMoves ^= i;
			URNonJumpMoves ^= i;
			LLNonJumpMoves ^= i;
			LRNonJumpMoves ^= i;
		}

		for (board_type i = 1; ULJumpMoves && URJumpMoves && LLJumpMoves && LRJumpMoves; i <<= 1)
		{
			if (i & ULJumpMoves)
			{

			}

			if (i & URJumpMoves)
			{

			}

			if (i & LLJumpMoves)
			{

			}

			if (i & LRJumpMoves)
			{

			}

			// update the moves
			ULJumpMoves ^= i;
			URJumpMoves ^= i;
			LLJumpMoves ^= i;
			LRJumpMoves ^= i;
		}

		return ret;
	}

	std::vector<BitBoard> BitBoard::GetPossibleWhiteMoves(BitBoard const &src)
	{

	}
}