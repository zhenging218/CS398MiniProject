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

#include "gpuminimax.h"

#define BLOCK_SIZE 32

#define RUN_RNG_CHECKERS 0

namespace
{
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

int main(int argc, char **argv)
{
	std::uint32_t max_depth = 4;
	std::uint32_t max_turns = 100;

	if (argc > 3)
	{
		std::cout << "usage: " << argv[0] << " [max_depth] [max_turns]\n";
		return 0;
	}
	else if (argc == 2)
	{
		max_depth = (std::uint32_t)atoi(argv[1]);
	}
	else if (argc == 3)
	{
		max_depth = (std::uint32_t)atoi(argv[1]);
		max_turns = (std::uint32_t)atoi(argv[2]);
	}
	
	std::cout << "search depth: " << max_depth << ", maximum game turns: " << max_turns << "\n";
    Checkers::Minimax::SetSearchDepth(max_depth);
    Checkers::Minimax::SetMaxTurns(max_turns);
	auto board = CreateRandomWhiteBitBoard();
	auto minimax = Checkers::CreateMinimaxBoard(board ,Checkers::Minimax::BLACK);
	// auto minimax = Checkers::CreateMinimaxBoard(Checkers::BitBoard(0xFFD10000u,0x00000FFFu,0u), Checkers::Minimax::BLACK);
	
	Checkers::Minimax::Result result = Checkers::Minimax::INPROGRESS;

	double totalTime = 0.0;
	double longestTime = std::numeric_limits<double>::min();
	double shortestTime = std::numeric_limits<double>::max();
	int turns = 0;

	while (result == Checkers::Minimax::INPROGRESS)
	{
		Checkers::Minimax::Turn turn = minimax.GetTurn();
		auto start = std::chrono::high_resolution_clock::now();
		result = minimax.Next();
		std::chrono::duration<double, std::milli> time = std::chrono::high_resolution_clock::now() - start;
		double dAvgSecs = time.count();

		std::cout << (turn == Checkers::Minimax::WHITE ? "White" : "Black") << ": ";

		if (result == Checkers::Minimax::LOSE)
		{
			std::cout << "Lost!\n";
		}
		else if (result == Checkers::Minimax::DRAW)
		{
			std::cout << "Draw game!\n";
		}
		else
		{
			std::cout << "Time taken for this turn's decision: " << dAvgSecs << " milliseconds\n";
			std::cout << minimax.GetBoard() << std::endl;
			totalTime += dAvgSecs;
			longestTime = std::max(longestTime, dAvgSecs);
			shortestTime = std::min(shortestTime, dAvgSecs);
			++turns;
		}
		
	}
	std::cout << "Total decisions made: " << turns << std::endl;
	std::cout << "Average time taken for each decision: " << (totalTime / (double)turns) << " milliseconds" << std::endl;
	std::cout << "Slowest decision took " << longestTime << " milliseconds\n";
	std::cout << "Fastest decision took " << shortestTime << " milliseconds\n";

	system("pause");
	system("cls");

	totalTime = 0.0;
	longestTime = std::numeric_limits<double>::min();
	shortestTime = std::numeric_limits<double>::max();
	turns = 0;
	int turns_left = Checkers::Minimax::GetMaxTurns();
	Checkers::Minimax::Result gpu_result = Checkers::Minimax::INPROGRESS;
	Checkers::Minimax::Turn gpu_turn = Checkers::Minimax::BLACK;
	const int gpu_depth = Checkers::Minimax::GetSearchDepth();

	while (gpu_result == Checkers::Minimax::INPROGRESS)
	{
		auto start = std::chrono::high_resolution_clock::now();
		gpu_result = Checkers::GPUMinimax::Next(board, gpu_turn, gpu_depth, turns_left);
		std::chrono::duration<double, std::milli> time = std::chrono::high_resolution_clock::now() - start;
		double dAvgSecs = time.count();

		std::cout << (gpu_turn == Checkers::Minimax::WHITE ? "White" : "Black") << ": ";

		if (gpu_result == Checkers::Minimax::LOSE)
		{
			std::cout << "Lost!\n";
		}
		else if (gpu_result == Checkers::Minimax::DRAW)
		{
			std::cout << "Draw game!\n";
		}
		else
		{
			std::cout << "Time taken for this turn's decision: " << dAvgSecs << " milliseconds\n";
			std::cout << board << std::endl;
			totalTime += dAvgSecs;
			longestTime = std::max(longestTime, dAvgSecs);
			shortestTime = std::min(shortestTime, dAvgSecs);
			++turns;
		}
	}
	std::cout << "Total decisions made: " << turns << std::endl;
	std::cout << "Average time taken for each decision: " << (totalTime / (double)turns) << " milliseconds" << std::endl;
	std::cout << "Slowest decision took " << longestTime << " milliseconds\n";
	std::cout << "Fastest decision took " << shortestTime << " milliseconds\n";

	system("PAUSE");
}