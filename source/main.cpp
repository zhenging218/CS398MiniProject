#include "precomp.h"
#include <stdlib.h>
#include <ctime>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
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
	
	std::cout << "testing bitboard init" << std::endl;
	Checkers::BitBoard bboard;
	std::cout << bboard << std::endl;

	auto moves = Checkers::BitBoard::GetPossibleBlackMoves(bboard);
	std::cout << "testing possible black moves" << std::endl;
	for (auto i : moves)
	{
		std::cout << i << std::endl;
		system("pause");
		system("cls");
	}

	system("PAUSE");
}