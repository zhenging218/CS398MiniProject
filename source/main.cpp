#include "precomp.h"
#include <stdlib.h>
#include <ctime>

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
	switch (temp)
	{
		case 1:board.Move(2, 0, Checkers::Board::Movement::BOTTOM_RIGHT);
			break;
		case 2:board.Move(2, 2, Checkers::Board::Movement::BOTTOM_LEFT);
			break;
		case 3:board.Move(2, 2, Checkers::Board::Movement::BOTTOM_RIGHT);
			break;
		case 4:board.Move(2, 4, Checkers::Board::Movement::BOTTOM_LEFT);
			break;
		case 5:board.Move(2, 4, Checkers::Board::Movement::BOTTOM_RIGHT);
			break;
		case 6:board.Move(2, 6, Checkers::Board::Movement::BOTTOM_LEFT);
			break;
		case 7:board.Move(2, 6, Checkers::Board::Movement::BOTTOM_RIGHT);
			break;

		/*case 1:board.Move(5, 1, Checkers::Board::Movement::TOP_RIGHT);
			break;
		case 2:board.Move(5, 3, Checkers::Board::Movement::TOP_LEFT);
			break;
		case 3:board.Move(5, 3, Checkers::Board::Movement::TOP_RIGHT);
			break;
		case 4:board.Move(5, 5, Checkers::Board::Movement::TOP_LEFT);
			break;
		case 5:board.Move(5, 5, Checkers::Board::Movement::TOP_RIGHT);
			break;
		case 6:board.Move(5, 7, Checkers::Board::Movement::TOP_LEFT);
			break;
		case 7:board.Move(5, 7, Checkers::Board::Movement::TOP_RIGHT);
			break;*/
	}
	
	std::cout << board << std::endl;
	system("PAUSE");
}