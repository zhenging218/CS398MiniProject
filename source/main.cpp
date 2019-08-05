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

	sdkDeleteTimer(&hTimer);

	system("PAUSE");
}