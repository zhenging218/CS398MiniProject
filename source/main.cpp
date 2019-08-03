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
	std::cout << bboard << std::endl;

	Checkers::Move moves[48];

	bool jumped = false;

	for (int x = 0; bboard.GetBlackPieceCount() && bboard.GetWhitePieceCount();)
	{
		if (x % 2)
		{
			std::vector<Checkers::Move> moves;
			auto start = bboard.GetPossibleWhiteMoves(std::back_insert_iterator<decltype(moves)>(moves));
			auto size = moves.size();
			if (size > 0)
			{
				if (jumped && !moves[0].jump)
				{
					jumped = false;
					std::cout << "new board is:\n";
					std::cout << bboard << std::endl;
					system("pause");
					++x;
					continue;
				}

				if (!jumped)
				{
					std::cout << "white move:\n";
					std::cout << size << " kinds of " << (moves[0].jump ? "jumps" : "moves") << " available\n";
				}

				std::uint32_t test = unif(rng);
				while (test >= size)
				{
					test = unif(rng);
				}
				bboard = moves[test].board;
				jumped = moves[test].jump;
				if (!jumped)
				{
					++x;
				}
				std::cout << "new board is:\n";
				std::cout << bboard << std::endl;
				system("pause");
			}
			else
			{
				std::cout << "white move:\n";
				std::cout << "no more moves\n";
				break;
			}

			
			//system("cls");
			
		}
		else
		{
			
			std::vector<Checkers::Move> moves;
			auto start = bboard.GetPossibleBlackMoves(std::back_insert_iterator<decltype(moves)>(moves));
			auto size = moves.size();
			if (size > 0)
			{
				if (jumped && !moves[0].jump)
				{
					jumped = false;
					std::cout << "new board is:\n";
					std::cout << bboard << std::endl;
					system("pause");
					++x;
					continue;
				}
				if (!jumped)
				{
					std::cout << "black move:\n";
					std::cout << size << " kinds of " << (moves[0].jump ? "jumps" : "moves") << " available\n";
				}
				

				std::uint32_t test = unif(rng);
				while (test >= size)
				{
					test = unif(rng);
				}
				bboard = moves[test].board;
				jumped = moves[test].jump;
				if (!jumped)
				{
					++x;
				}
				std::cout << "new board is:\n";
				std::cout << bboard << std::endl;
				system("pause");
			}
			else
			{
				std::cout << "black move:\n";
				std::cout << "no more moves\n";
				break;
			}

			
			//system("cls");
			//system("cls");
			
		}
	}

	system("PAUSE");
}