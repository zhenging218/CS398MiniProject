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

	void RunCPUVersion(Checkers::BitBoard const &board, Checkers::Minimax::Turn start_turn, double &shortestTime, double &longestTime, double &average, int &turns, bool show_game)
	{
		auto minimax = Checkers::CreateMinimaxBoard(board, start_turn);

		double totalTime = 0.0;
		longestTime = std::numeric_limits<double>::min();
		shortestTime = std::numeric_limits<double>::max();
		turns = 0;

		Checkers::Minimax::Result result = Checkers::Minimax::INPROGRESS;
		while (result == Checkers::Minimax::INPROGRESS)
		{
			Checkers::Minimax::Turn turn = minimax.GetTurn();
			auto start = std::chrono::high_resolution_clock::now();
			result = minimax.Next();
			std::chrono::duration<double, std::milli> time = std::chrono::high_resolution_clock::now() - start;
			double dAvgSecs = time.count();

			if (show_game)
			{
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
				system("pause");
			}
			else
			{
				totalTime += dAvgSecs;
				longestTime = std::max(longestTime, dAvgSecs);
				shortestTime = std::min(shortestTime, dAvgSecs);
				++turns;
			}

		}

		average = totalTime / (double)turns;
	}

	void RunGPUVersion(Checkers::BitBoard const &src_board, Checkers::Minimax::Turn start_turn, double &shortestTime, double &longestTime, double &average, int &turns, bool show_game)
	{
		double totalTime = 0.0;
		longestTime = std::numeric_limits<double>::min();
		shortestTime = std::numeric_limits<double>::max();
		turns = 0;

		Checkers::Minimax::Result gpu_result = Checkers::Minimax::INPROGRESS;
		Checkers::BitBoard board = src_board;
		Checkers::Minimax::Turn gpu_turn = start_turn;
		int turns_left = Checkers::Minimax::GetMaxTurns();
		const int gpu_depth = Checkers::Minimax::GetSearchDepth();

		while (gpu_result == Checkers::Minimax::INPROGRESS)
		{
			if (show_game)
			{
				std::cout << (gpu_turn == Checkers::Minimax::WHITE ? "white turn" : "black turn") << "\n";
			}
			auto start = std::chrono::high_resolution_clock::now();
			gpu_result = Checkers::GPUMinimax::Next(board, gpu_turn, gpu_depth, turns_left);
			
			std::chrono::duration<double, std::milli> time = std::chrono::high_resolution_clock::now() - start;
			double dAvgSecs = time.count();

			if (show_game)
			{
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
				system("pause");
			}
			else
			{
				totalTime += dAvgSecs;
				longestTime = std::max(longestTime, dAvgSecs);
				shortestTime = std::min(shortestTime, dAvgSecs);
				++turns;
			}
		}

		average = totalTime / (double)turns;
		
	}

	void BenchBoth(Checkers::BitBoard const &src_board, Checkers::Minimax::Turn start_turn)
	{
		int progress = 0;

		double cpu_totalTime = 0.0;
		double cpu_shortestTime = std::numeric_limits<double>::max();
		double cpu_longestTime = std::numeric_limits<double>::min();
		double cpu_average;
		int cpu_turns = 0;
		int slowest_cpu_turn = 0;
		int fastest_cpu_turn = 0;
		Checkers::Minimax::Result cpu_result = Checkers::Minimax::INPROGRESS;
		Checkers::Minimax::Turn cpu_turn = start_turn;

		double gpu_totalTime = 0.0;
		double gpu_shortestTime = std::numeric_limits<double>::max();
		double gpu_longestTime = std::numeric_limits<double>::min();
		double gpu_average;
		int gpu_turns = 0;
		int slowest_gpu_turn = 0;
		int fastest_gpu_turn = 0;
		Checkers::Minimax::Result gpu_result = Checkers::Minimax::INPROGRESS;
		Checkers::Minimax::Turn gpu_turn = start_turn;

		std::vector<Checkers::BitBoard> cpu_out, gpu_out;

		auto minimax = Checkers::CreateMinimaxBoard(src_board, start_turn);

		while (cpu_result == Checkers::Minimax::INPROGRESS)
		{
			std::cout << "CPU Version: Turn " << cpu_turns << "...\r";
			Checkers::Minimax::Turn cpu_turn = minimax.GetTurn();
			auto start = std::chrono::high_resolution_clock::now();
			cpu_result = minimax.Next();
			std::chrono::duration<double, std::milli> time = std::chrono::high_resolution_clock::now() - start;
			double dAvgSecs = time.count();
			if(cpu_result != Checkers::Minimax::LOSE && cpu_result != Checkers::Minimax::DRAW)
			{
				cpu_out.emplace_back(minimax.GetBoard());
				cpu_totalTime += dAvgSecs;
				if (dAvgSecs > cpu_longestTime)
				{
					cpu_longestTime = dAvgSecs;
					slowest_cpu_turn = cpu_turns;
				}
				else if (dAvgSecs < cpu_shortestTime)
				{
					cpu_shortestTime = dAvgSecs;
					fastest_cpu_turn = cpu_turns;
				}
				++cpu_turns;
			}
		}

		cpu_average = cpu_totalTime / (double)cpu_turns;

		std::cout << "CPU version run complete\n";

		Checkers::BitBoard gpu_board = src_board;
		
		int turns_left = Checkers::Minimax::GetMaxTurns();
		const int gpu_depth = Checkers::Minimax::GetSearchDepth();
		progress = 0;

		while (gpu_result == Checkers::Minimax::INPROGRESS)
		{
			std::cout << "GPU Version: Turn " << gpu_turns << "...\r";
			auto start = std::chrono::high_resolution_clock::now();
			gpu_result = Checkers::GPUMinimax::Next(gpu_board, gpu_turn, gpu_depth, turns_left);
			std::chrono::duration<double, std::milli> time = std::chrono::high_resolution_clock::now() - start;
			double dAvgSecs = time.count();

			if (gpu_result != Checkers::Minimax::LOSE && gpu_result != Checkers::Minimax::DRAW)
			{
				gpu_out.emplace_back(gpu_board);
				gpu_totalTime += dAvgSecs;
				if (dAvgSecs > gpu_longestTime)
				{
					gpu_longestTime = dAvgSecs;
					slowest_gpu_turn = gpu_turns;
				}
				else if (dAvgSecs < gpu_shortestTime)
				{
					gpu_shortestTime = dAvgSecs;
					fastest_gpu_turn = gpu_turns;
				}
				++gpu_turns;
			}
		}

		gpu_average = gpu_totalTime / (gpu_turns);

		std::cout << "GPU version run complete\n";
		
		std::cout << "\n\n";

		std::cout << "Showing results:\n";

		std::cout << "CPU: took " << cpu_turns << " turns and ran for " << cpu_totalTime << " milliseconds in total\n";
		std::cout << "The game result was " << (cpu_result == Checkers::Minimax::DRAW ? "a draw game." : (cpu_turn == Checkers::Minimax::WHITE ? "black won." : "white won.")) << "\n";
		std::cout << "Average time taken for each decision: " << cpu_average << " milliseconds\n";
		std::cout << "Slowest decision took " << cpu_longestTime << " milliseconds at turn " << slowest_cpu_turn << "\n";
		std::cout << "Fastest decision took " << cpu_shortestTime << " milliseconds at turn " << fastest_cpu_turn << "\n";

		std::cout << "\n\n";

		std::cout << "GPU: took " << gpu_turns << " turns and ran for " << gpu_totalTime << " milliseconds in total\n";
		std::cout << "The game result was " << (gpu_result == Checkers::Minimax::DRAW ? "a draw game." : (gpu_turn == Checkers::Minimax::WHITE ? "black won." : "white won.")) << "\n";
		std::cout << "Average time taken for each decision: " << gpu_average << " milliseconds\n";
		std::cout << "Slowest decision took " << gpu_longestTime << " milliseconds at turn " << slowest_gpu_turn << "\n";
		std::cout << "Fastest decision took " << gpu_shortestTime << " milliseconds at turn " << fastest_gpu_turn << "\n";

		std::cout << "\n\n";

		std::cout << "Testing correctness...\n";
		bool decisions_ok = cpu_turns == gpu_turns && cpu_out.size() == gpu_out.size();
		if (decisions_ok)
		{
			bool board_ok = true;
			auto size = cpu_out.size();
			for (decltype(size) i = 0; i < size && board_ok; ++i)
			{
				board_ok = cpu_out[i] == gpu_out[i];
			}

			if (board_ok)
			{
				std::cout << "CPU and GPU ran the same game!\n";
			}
			else
			{
				std::cout << "CPU and GPU games had same amount of turns but moves did not match!\n";
			}
		}
		else
		{
			std::cout << "CPU and GPU games did not have the same amount of turns!\n";
		}
	}
}

int main(int argc, char **argv)
{
	std::uint32_t max_depth = 4;
	std::uint32_t max_turns = 100;
	int run_which = 0;

	if (argc > 4)
	{
		std::cout << "usage: " << argv[0] << "[cpu:0, cpu_benchmark:1, gpu:2, gpu_benchmark:3, benchmark:4] [max_depth] [max_turns]\n";
		return 0;
	}

	run_which = atoi(argv[1]);
	if (run_which < 0 || run_which > 4)
	{
		std::cout << "usage: " << argv[0] << "[cpu:0, cpu_benchmark:1, gpu:2, gpu_benchmark:3, benchmark:4] [max_depth] [max_turns]\n";
		return 0;
	}

	if (argc == 3)
	{
		max_depth = (std::uint32_t)atoi(argv[2]);
	}
	else if (argc == 4)
	{
		max_depth = (std::uint32_t)atoi(argv[2]);
		max_turns = (std::uint32_t)atoi(argv[3]);
	}

	std::cout << "search depth: " << max_depth << ", maximum game turns: " << max_turns << "\n";
	Checkers::Minimax::SetSearchDepth(max_depth);
	Checkers::Minimax::SetMaxTurns(max_turns);

	auto board = CreateRandomWhiteBitBoard();
	// auto board = Checkers::BitBoard(0xFFD10000u, 0x00000FFFu, 0u);

	double longestTime;
	double shortestTime;
	double average;
	int turns;

	switch (run_which)
	{
	case 0:
		RunCPUVersion(board, Checkers::Minimax::BLACK, shortestTime, longestTime, average, turns, true);
		std::cout << "Average time taken for each decision: " << average << " milliseconds" << std::endl;
		std::cout << "Slowest decision took " << longestTime << " milliseconds\n";
		std::cout << "Fastest decision took " << shortestTime << " milliseconds\n";
		break;
	case 1:
		RunCPUVersion(board, Checkers::Minimax::BLACK, shortestTime, longestTime, average, turns, false);
		std::cout << "Average time taken for each decision: " << average << " milliseconds" << std::endl;
		std::cout << "Slowest decision took " << longestTime << " milliseconds\n";
		std::cout << "Fastest decision took " << shortestTime << " milliseconds\n";
		break;
	case 2:
		RunGPUVersion(board, Checkers::Minimax::BLACK, shortestTime, longestTime, average, turns, true);
		std::cout << "Average time taken for each decision: " << average << " milliseconds" << std::endl;
		std::cout << "Slowest decision took " << longestTime << " milliseconds\n";
		std::cout << "Fastest decision took " << shortestTime << " milliseconds\n";
		break;
	case 3:
		RunGPUVersion(board, Checkers::Minimax::BLACK, shortestTime, longestTime, average, turns, false);
		std::cout << "Average time taken for each decision: " << average << " milliseconds" << std::endl;
		std::cout << "Slowest decision took " << longestTime << " milliseconds\n";
		std::cout << "Fastest decision took " << shortestTime << " milliseconds\n";
		break;
	case 4:
		BenchBoth(board, Checkers::Minimax::BLACK);
		break;
	default:
		ASSERT(0, "run_which value is wrong (%d)!", run_which);
		break;
	}

	system("PAUSE");
}