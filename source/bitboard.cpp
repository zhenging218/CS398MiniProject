#include "precomp.h"
#include <utility>
#include <array>

#define UPPER_RIGHT(x) (((x) & (~InvalidURMove)) << 3)
#define LOWER_RIGHT(x) (((x) & (~InvalidLRMove)) >> 4)
#define UPPER_LEFT(x)  (((x) & (~InvalidULMove)) << 4)
#define LOWER_LEFT(x)  (((x) & (~InvalidLLMove)) >> 3)

namespace Checkers
{
	namespace
	{
		constexpr BitBoard::board_type KingMask = 0xF000000F;
		constexpr BitBoard::board_type Mask = 0xFFFFFFFF;

		constexpr BitBoard::board_type InvalidULMove = 0xF0808080;
		constexpr BitBoard::board_type InvalidLLMove = 0x0080808F;
		constexpr BitBoard::board_type InvalidURMove = 0xF1010100;
		constexpr BitBoard::board_type InvalidLRMove = 0x0101010F;

		BitBoard::board_type InitializeWhitePieces()
		{
			return 0xFFF00000;
		}

		BitBoard::board_type InitializeBlackPieces()
		{
			return 0x00000FFF;
		}
	}

	BitBoard::BitBoard() : white(InitializeWhitePieces()), black(InitializeBlackPieces()), kings(0)
	{

	}

	BitBoard::BitBoard(BitBoard::board_type w, BitBoard::board_type b, BitBoard::board_type k) : white(w), black(b), kings(k)
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

	BitBoard::board_type BitBoard::GetURNonJumpWhiteMoves() const
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

	BitBoard::board_type BitBoard::GetURJumpWhiteMoves() const
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

	BitBoard::board_type BitBoard::GetWhiteKings() const
	{
		return kings & white;
	}

	BitBoard::board_type BitBoard::GetBlackKings() const
	{
		return kings & black;
	}

	BitBoard::board_type BitBoard::GetBlackPieces() const
	{
		return black;
	}

	BitBoard::board_type BitBoard::GetWhitePieces() const
	{
		return white;
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

		for (board_type i = 1; i && (ULNonJumpMoves || URNonJumpMoves || LLNonJumpMoves || LRNonJumpMoves); i <<= 1)
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
			ULNonJumpMoves &= ~i;
			URNonJumpMoves &= ~i;
			LLNonJumpMoves &= ~i;
			LRNonJumpMoves &= ~i;
		}

		for (board_type i = 1; i && (ULJumpMoves || URJumpMoves || LLJumpMoves || LRJumpMoves); i <<= 1)
		{
			if (i & ULJumpMoves)
			{
				ret.push_back(BitBoard(
					src.white ^ UPPER_LEFT(i),
					(src.black & (Mask ^ i)) | UPPER_LEFT(UPPER_LEFT(i)),
					((i & src.kings) ^ src.kings) & (UPPER_LEFT(UPPER_LEFT(i)) & KingMask)
				));
			}

			if (i & URJumpMoves)
			{
				ret.push_back(BitBoard(
					src.white ^ UPPER_RIGHT(i),
					(src.black & (Mask ^ i)) | UPPER_RIGHT(UPPER_RIGHT(i)),
					((i & src.kings) ^ src.kings) & (UPPER_RIGHT(UPPER_RIGHT(i)) & KingMask)
				));
			}

			if (i & LLJumpMoves)
			{
				ret.push_back(BitBoard(
					src.white ^ LOWER_LEFT(i),
					(src.black & (Mask ^ i)) | LOWER_LEFT(LOWER_LEFT(i)),
					((i & src.kings) ^ src.kings) & (LOWER_LEFT(LOWER_LEFT(i)) & KingMask)
				));
			}

			if (i & LRJumpMoves)
			{
				ret.push_back(BitBoard(
					src.white ^ LOWER_RIGHT(i),
					(src.black & (Mask ^ i)) | LOWER_RIGHT(LOWER_RIGHT(i)),
					((i & src.kings) ^ src.kings) & (LOWER_RIGHT(LOWER_RIGHT(i)) & KingMask)
				));
			}

			// update the moves
			ULJumpMoves &= ~i;
			URJumpMoves &= ~i;
			LLJumpMoves &= ~i;
			LRJumpMoves &= ~i;
		}

		return ret;
	}

	std::vector<BitBoard> BitBoard::GetPossibleWhiteMoves(BitBoard const &src)
	{
		std::vector<BitBoard> ret;

		board_type ULNonJumpMoves = src.GetULNonJumpWhiteMoves();
		board_type URNonJumpMoves = src.GetURNonJumpWhiteMoves();
		board_type LLNonJumpMoves = src.GetLLNonJumpWhiteMoves();
		board_type LRNonJumpMoves = src.GetLRNonJumpWhiteMoves();

		board_type ULJumpMoves = src.GetULJumpWhiteMoves();
		board_type URJumpMoves = src.GetURJumpWhiteMoves();
		board_type LLJumpMoves = src.GetLLJumpWhiteMoves();
		board_type LRJumpMoves = src.GetLRJumpWhiteMoves();

		for (board_type i = 1; i && (ULNonJumpMoves || URNonJumpMoves || LLNonJumpMoves || LRNonJumpMoves); i <<= 1)
		{
			if (i & ULNonJumpMoves)
			{
				ret.push_back(BitBoard(
					(src.white & (Mask ^ i)) | UPPER_LEFT(i),
					src.black,
					((i & src.kings) ^ src.kings) & (UPPER_LEFT(i) & KingMask)
				));
			}

			if (i & URNonJumpMoves)
			{
				ret.push_back(BitBoard(
					(src.white & (Mask ^ i)) | UPPER_RIGHT(i),
					src.black,
					(i & src.kings) ^ src.kings & (UPPER_RIGHT(i) & KingMask)
				));
			}

			if (i & LLNonJumpMoves)
			{
				ret.push_back(BitBoard(
					(src.white & (Mask ^ i)) | LOWER_LEFT(i),
					src.black,
					(i & src.kings) ^ src.kings & (LOWER_LEFT(i) & KingMask)
				));
			}

			if (i & LRNonJumpMoves)
			{
				ret.push_back(BitBoard(
					(src.white & (Mask ^ i)) | LOWER_RIGHT(i),
					src.black,
					(i & src.kings) ^ src.kings & (LOWER_RIGHT(i) & KingMask)
				));
			}

			// update the moves
			ULNonJumpMoves &= ~i;
			URNonJumpMoves &= ~i;
			LLNonJumpMoves &= ~i;
			LRNonJumpMoves &= ~i;
		}

		for (board_type i = 1; i && (ULJumpMoves || URJumpMoves || LLJumpMoves || LRJumpMoves); i <<= 1)
		{
			if (i & ULJumpMoves)
			{
				ret.push_back(BitBoard(
					(src.white & (Mask ^ i)) | UPPER_LEFT(UPPER_LEFT(i)),
					src.black ^ UPPER_LEFT(i),
					((i & src.kings) ^ src.kings) & (UPPER_LEFT(UPPER_LEFT(i)) & KingMask)
				));
			}

			if (i & URJumpMoves)
			{
				ret.push_back(BitBoard(
					(src.white & (Mask ^ i)) | UPPER_RIGHT(UPPER_RIGHT(i)),
					src.black ^ UPPER_RIGHT(i),
					((i & src.kings) ^ src.kings) & (UPPER_RIGHT(UPPER_RIGHT(i)) & KingMask)
				));
			}

			if (i & LLJumpMoves)
			{
				ret.push_back(BitBoard(
					(src.white & (Mask ^ i)) | LOWER_LEFT(LOWER_LEFT(i)),
					src.black ^ LOWER_LEFT(i),
					((i & src.kings) ^ src.kings) & (LOWER_LEFT(LOWER_LEFT(i)) & KingMask)
				));
			}

			if (i & LRJumpMoves)
			{
				ret.push_back(BitBoard(
					(src.white & (Mask ^ i)) | LOWER_RIGHT(LOWER_RIGHT(i)),
					src.black ^ LOWER_RIGHT(i),
					((i & src.kings) ^ src.kings) & (LOWER_RIGHT(LOWER_RIGHT(i)) & KingMask)
				));
			}

			// update the moves
			ULJumpMoves &= ~i;
			URJumpMoves &= ~i;
			LLJumpMoves &= ~i;
			LRJumpMoves &= ~i;
		}

		return ret;
	}

	std::ostream &operator<<(std::ostream &os, BitBoard const &src)
	{
		using board_type = BitBoard::board_type;

		board_type white = src.GetWhitePieces();
		board_type black = src.GetBlackPieces();
		board_type white_kings = src.GetWhiteKings();
		board_type black_kings = src.GetBlackKings();

		int start = 0;
		int x = 1;

		for(int j = 0; j < 8; ++j)
		{
			os << "|";
			for (int i = 0; i < 8; ++i)
			{	
				// check if index is even or odd
				if (start)
				{
					// odd
					if (i % 2)
					{
						os << ((x & white) ? (x & (white_kings) ? "XX" : " X") : ((x & black) ? (x & black_kings) ? "OO" : " O" : "  ")) << "|";
						x <<= 1;
					}
					else
					{
						os << "  |";
					}
				}
				else
				{
					// even
					if (!(i % 2))
					{
						os << ((x & white) ? (x & (white_kings) ? "XX" : " X") : ((x & black) ? (x & black_kings) ? "OO" : " O" : "  ")) << "|";
						x <<= 1;
					}
					else
					{
						os << "  |";
					}
				}
			}
			os << "\n";
			start = 1 - start;
		}

		return os;
	}
}