#pragma once

#ifdef _DEBUG

#include <iostream>
#include <cstdio>
#include <exception>

// suppress unnamed struct warning
#pragma warning(disable : 4996)

#define ASSERT(expression, fmt, ...)\
do\
{\
	if(!(expression))\
	{\
		char buf[256];\
		char buf2[256];\
		sprintf(buf, fmt, __VA_ARGS__);\
		sprintf(buf2, "Assert failed at %s (line %u):\n\n\t%s\n\nMessage: %s", __FILE__, __LINE__, #expression, buf);\
		std::cerr << buf2 << std::endl;\
		throw(std::runtime_error(buf2));\
	}\
} while(0)

#else

#define ASSERT(expression, fmt, ...)

#endif

#define UNUSED_VARIABLE(x) (void)x