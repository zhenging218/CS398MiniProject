#include "precomp.h"
#include <stdlib.h>
#include <ctime>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <random>
#include <chrono>
#include <iterator>

#define BLOCK_SIZE 32

#define RUN_RNG_CHECKERS 0

namespace
{
	std::pair<std::vector<Checkers::Move>, bool> TestGetMoves(Checkers::BitBoard const &board, Checkers::Minimax::Turn turn)
	{
		std::vector<Checkers::Move> ret;
		bool jumped = false;
		switch (turn)
		{
		case Checkers::Minimax::Turn::WHITE:
		{
			auto result = board.GetPossibleWhiteMoves(std::back_insert_iterator<decltype(ret)>(ret));
			jumped = result.second;
		} break;
		case Checkers::Minimax::Turn::BLACK:
		{
			auto result = board.GetPossibleBlackMoves(std::back_insert_iterator<decltype(ret)>(ret));
			jumped = result.second;
		} break;
		default:
			ASSERT(0, "GetMoves(%u) has wrong turn value!", turn);
			break;
		}
		return std::make_pair(ret, jumped);
	}

	Checkers::BitBoard CreateRandomWhiteBitBoard()
	{
		std::mt19937_64 rng;
		// initialize the random number generator with time-dependent seed
		uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
		std::seed_seq ss{ uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> 32) };
		rng.seed(ss);
		std::uniform_int_distribution<int> unif(16, 19);
		std::uniform_int_distribution<int> unif2(3, 4);
		int pos = unif(rng);
		int del = unif2(rng);
		if (pos == 16)
		{
			del = 4;
		}
		return Checkers::BitBoard((0xFFF00000u & ~((1u << pos) << del)) | (1u << pos), 0x00000FFFu, 0u);
	}
}

int RandomStart()
{
	/*
	| X |   | X |   | X |   | X |   |
	|   | X |   | X |   | X |   | X |
	| X |   | X |   | X |   | X |   |
	|   |1,2|   |3,4|   |5,6|   | 7 |
	|   |   |   |   |   |   |   |   |
	|   | O |   | O |   | O |   | O |
	| O |   | O |   | O |   | O |   |
	|   | O |   | O |   | O |   | O |
	*/
	//1 to 7
	return (rand() % 7 + 1);
}

int main()
{
	srand((unsigned int)time(0));
	Checkers::Board board = Checkers::Board();
	int temp = RandomStart();
	std::cout << "Random Start number: "<< temp << std::endl;
	//switch (temp)
	//{
	//	case 1:board.Move(2, 0, Checkers::Board::Movement::BOTTOM_RIGHT);
	//		break;
	//	case 2:board.Move(2, 2, Checkers::Board::Movement::BOTTOM_LEFT);
	//		break;
	//	case 3:board.Move(2, 2, Checkers::Board::Movement::BOTTOM_RIGHT);
	//		break;
	//	case 4:board.Move(2, 4, Checkers::Board::Movement::BOTTOM_LEFT);
	//		break;
	//	case 5:board.Move(2, 4, Checkers::Board::Movement::BOTTOM_RIGHT);
	//		break;
	//	case 6:board.Move(2, 6, Checkers::Board::Movement::BOTTOM_LEFT);
	//		break;
	//	case 7:board.Move(2, 6, Checkers::Board::Movement::BOTTOM_RIGHT);
	//		break;

	//	/*case 1:board.Move(5, 1, Checkers::Board::Movement::TOP_RIGHT);
	//		break;
	//	case 2:board.Move(5, 3, Checkers::Board::Movement::TOP_LEFT);
	//		break;
	//	case 3:board.Move(5, 3, Checkers::Board::Movement::TOP_RIGHT);
	//		break;
	//	case 4:board.Move(5, 5, Checkers::Board::Movement::TOP_LEFT);
	//		break;
	//	case 5:board.Move(5, 5, Checkers::Board::Movement::TOP_RIGHT);
	//		break;
	//	case 6:board.Move(5, 7, Checkers::Board::Movement::TOP_LEFT);
	//		break;
	//	case 7:board.Move(5, 7, Checkers::Board::Movement::TOP_RIGHT);
	//		break;*/
	//}
	
	/*
	StopWatchInterface *hTimer = NULL;
	float dAvgSecs;
	for (int i = 0; i < 5; ++i)
	{
		std::cout << "Checkers starting..." << std::endl;
		std::cout << std::endl;
		
		std::cout << board << std::endl;
		sdkCreateTimer(&hTimer);

		sdkResetTimer(&hTimer);
		sdkStartTimer(&hTimer);
		std::cout << std::endl;
		std::cout << "...allocating GPU memory" << std::endl;

		sdkStopTimer(&hTimer);
		dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer);
		std::cout << "Time taken for GPU minimax" << std::endl;

		sdkResetTimer(&hTimer);
		sdkStartTimer(&hTimer);
		std::cout << std::endl;
		std::cout << "...allocating CPU memory" << std::endl;
		// Start part of CPU minmax algorithm

		sdkStopTimer(&hTimer);
		dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer);
		std::cout << "Time taken for CPU minimax" << std::endl;
		Sleep(5000);
		system("cls");
	}
	*/
	
#if RUN_RNG_CHECKERS
	std::mt19937_64 rng;
	// initialize the random number generator with time-dependent seed
	uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	std::seed_seq ss{ uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> 32) };
	rng.seed(ss);
	// initialize a uniform distribution between 0 and 1
	std::uniform_int_distribution<int> unif(0, 48);

	std::cout << "testing bitboard init" << std::endl;
	// Checkers::BitBoard bboard(0x00040000u, 0x00002000u, 0x00002000u);
	Checkers::BitBoard bboard;
	std::cout << board << std::endl;

	Checkers::Move moves[48];

	bool jumped = false;

	for (int x = 0; bboard.GetBlackPieceCount() && bboard.GetWhitePieceCount();)
	{
		if (!(x % 2))
		{
			std::cout << "White Turn (PLAYER 1):\n";
			std::cout << "White has " << bboard.GetWhitePieceCount() << " pieces (" << bboard.GetWhiteKingsCount() << " kings)\n";
			auto moves = TestGetMoves(bboard, Checkers::Minimax::Turn::WHITE);
			std::vector<Checkers::Move> frontier;

			if (moves.second)
			{
				while (!moves.first.empty())
				{
					std::vector<Checkers::Move> moves2;
					for (auto i = moves.first.begin(); i != moves.first.end(); ++i)
					{

						auto j = TestGetMoves(i->board, Checkers::Minimax::Turn::WHITE);
						if (j.second)
						{
							moves2.insert(moves2.end(), j.first.begin(), j.first.end());
						}
						else
						{
							frontier.push_back(*i);
						}
					}
					moves.first = std::move(moves2);
				}
			}
			else
			{
				frontier = std::move(moves.first);
			}

			if (!frontier.empty())
			{
				std::uint32_t test = unif(rng);
				while (test >= frontier.size())
				{
					test = unif(rng);
				}
				bboard = frontier[test].board;
				std::cout << bboard << std::endl;
				system("pause");
			}
			else
			{
				std::cout << "white move:\n";
				std::cout << "no more moves\n";
				break;
			}
		}
		else
		{
			std::cout << "Black Turn (PLAYER 2):\n";
			std::cout << "Black has " << bboard.GetBlackPieceCount() << " pieces (" << bboard.GetBlackKingsCount() << " kings)\n";
			auto moves = TestGetMoves(bboard, Checkers::Minimax::Turn::BLACK);
			std::vector<Checkers::Move> frontier;

			if (moves.second)
			{
				while (!moves.first.empty())
				{
					std::vector<Checkers::Move> moves2;
					for (auto i = moves.first.begin(); i != moves.first.end(); ++i)
					{

						auto j = TestGetMoves(i->board, Checkers::Minimax::Turn::BLACK);
						if (j.second)
						{
							moves2.insert(moves2.end(), j.first.begin(), j.first.end());
						}
						else
						{
							frontier.push_back(*i);
						}
					}
					moves.first = std::move(moves2);
				}
			}
			else
			{
				frontier = std::move(moves.first);
			}

			if (!frontier.empty())
			{
				std::uint32_t test = unif(rng);
				while (test >= frontier.size())
				{
					test = unif(rng);
				}
				bboard = frontier[test].board;
				std::cout << bboard << std::endl;
				system("pause");
			}
			else
			{
				std::cout << "black move:\n";
				std::cout << "no more moves\n";
				break;
			}
		}
		++x;
	}

	if (bboard.GetBlackPieceCount() == 0)
	{
		std::cout << "White (PLAYER 1) Wins!\n";
	}
	else if (bboard.GetWhitePieceCount() == 0)
	{
		std::cout << "Black (PLAYER 2) Wins!\n";
	}
	else
	{
		std::cout << "Draw!\n" << std::endl;
	}
#else

system("cls");
	auto minimax = Checkers::CreateMinimaxBoard(CreateRandomWhiteBitBoard());
	Checkers::Minimax::Result result = Checkers::Minimax::INPROGRESS;
	while (result == Checkers::Minimax::INPROGRESS)
	{
		Checkers::Minimax::Turn turn = minimax.GetTurn();
		result = minimax.Next();
		std::cout << (turn == Checkers::Minimax::WHITE ? "White" : "Black") << ":\n";
		if (result == Checkers::Minimax::LOSE)
		{
			std::cout << (turn == Checkers::Minimax::WHITE ? "White" : "Black") << " lost!\n";
		}
		else
		{
			std::cout << minimax.GetBoard() << std::endl;
		}
	}
#endif

	system("PAUSE");
}