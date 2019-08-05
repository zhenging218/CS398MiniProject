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
	std::pair<std::vector<Checkers::BitBoard>, bool> TestGetMoves(Checkers::BitBoard const &board, Checkers::Minimax::Turn turn)
	{
		std::vector<Checkers::BitBoard> ret;
		bool jumped = false;
		switch (turn)
		{
		case Checkers::Minimax::Turn::WHITE:
		{
			jumped = Checkers::BitBoard::GetPossibleWhiteMoves(board, std::back_insert_iterator<decltype(ret)>(ret));
		} break;
		case Checkers::Minimax::Turn::BLACK:
		{
			jumped = Checkers::BitBoard::GetPossibleBlackMoves(board, std::back_insert_iterator<decltype(ret)>(ret));
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
	int temp = RandomStart();
	std::cout << "Random Start number: "<< temp << std::endl;
	
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
    Checkers::Minimax::SetSearchDepth(4);
    Checkers::Minimax::SetMaxTurns(100);
	//auto minimax = Checkers::CreateMinimaxBoard(CreateRandomWhiteBitBoard(),Checkers::Minimax::BLACK);
	auto minimax = Checkers::CreateMinimaxBoard(Checkers::BitBoard(0xFFD10000u,0x00000FFFu,0u), Checkers::Minimax::BLACK);
	
	Checkers::Minimax::Result result = Checkers::Minimax::INPROGRESS;
	

	StopWatchInterface *hTimer = NULL;
	sdkCreateTimer(&hTimer);

	double totalTime = 0.0;
	double longestTime = std::numeric_limits<double>::min();
	double shortestTime = std::numeric_limits<double>::max();
	int turns = 0;

	while (result == Checkers::Minimax::INPROGRESS)
	{
		Checkers::Minimax::Turn turn = minimax.GetTurn();
		std::cout << (turn == Checkers::Minimax::WHITE ? "White" : "Black") << ":\n";
		sdkResetTimer(&hTimer);
		sdkStartTimer(&hTimer);
		result = minimax.Next();
		std::cout << minimax.GetBoard() << std::endl;
		sdkStopTimer(&hTimer);
		double dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer);
		
		if (result == Checkers::Minimax::LOSE)
		{
			std::cout << (turn == Checkers::Minimax::WHITE ? "White" : "Black") << " lost!\n";
		}
		else if (result == Checkers::Minimax::DRAW)
		{
			std::cout << "Draw game!\n";
		}
		else
		{
			std::cout << "Time taken for this turn's decision: " << dAvgSecs << " seconds" << std::endl;
			totalTime += dAvgSecs;
			longestTime = std::max(longestTime, dAvgSecs);
			shortestTime = std::min(shortestTime, dAvgSecs);
			++turns;
		}
	}
	std::cout << "Total decisions made: " << turns << std::endl;
	std::cout << "Average time taken for each decision: " << (totalTime / (double)turns) << " seconds" << std::endl;
	std::cout << "Slowest decision took " << longestTime << " seconds\n";
	std::cout << "Fastest decision took " << shortestTime << " seconds\n";
#endif

	sdkDeleteTimer(&hTimer);

	system("PAUSE");
}