/****************************************************************** /
/*!
\file READNE
\author Ng Zi Jian, Candice Chong Fang Qi, Yiu Zhen Ging
\par zijian.ng@digipen.edu
\date 16-8-2019
\brief
Copyright (C) 2019 DigiPen Institute of Technology.
Reproduction or disclosure of this file or its contents without the
prior written consent of DigiPen Institute of Technology is prohibited.
*/
/* End Header
*******************************************************************/

1) Open Visual Studio 2015.
2) Press Ctrl Shift B or Build TAB->Build Solution to compile once in visual studio 2015.
3) Open Cygwin.
4) Navigate to the release directory (..\CS398MiniProject\build\x64\Release) once compile is finished.
5) Usage is as of following ./checkers.exe [Option 1] [Option 2] [Option 3]
   [Option 1]
   0 = Display movements using CPU minimax 
   1 = Display only time taken in CPU
   2 = Display movements using GPU minimax 
   3 = Display only time taken in GPU, 
   4 = Display both GPU and CPU movements, 
   5 = Display both GPU and CPU timing 
   6 = GPU as black vs CPU as white
   [Option 2: Depth]
   Depth to cut off
   [Option 3: Turns]
   Turns to cut off

How to read board on the screen:
|  | W|  | W|  | W|  | W|
| W|  | W|  | W|  | W|  |
|  |  |  | W|  | W|  | W|
| W|  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
| B|  | B|  | B|  | B|  |
|  | B|  | B|  | B|  | B|
| B|  | B|  | B|  | B|  |
W - white player piece
B - black player piece
WW - white player king piece
BB - black player king piece

New boards will be printed on screen after each move:
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  | B|  |
| W|  |  |  |  |  |  |  |
|  |  |  |  |BB|  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |WW|  |

Submission structure:
docs - Contains presentation slides and report
ref - source used to implement
source - all the .cu,.cpp,.h are at 