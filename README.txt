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
   cpu:0 = Display movements using CPU minimax 
   cpu_benchmark:1 = Display only time taken in CPU
   gpu:2 = Display movements using GPU minimax 
   gpu_benchmark:3 = Display only time taken in GPU, 
   benchmark:4 = Display both GPU and CPU movements, 
   lean_benchmark:5 = Display both GPU and CPU timing 
   fight:6 = GPU as black vs CPU as white
   [Option 2: Depth]
   Depth to cut off
   [Option 3: Turns]
   Turns to cut off