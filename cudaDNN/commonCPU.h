#pragma once
#include <iostream>
#include <string>
#define LINE_WIDTH 50

namespace miduho
{
	inline void printLine()
	{
		std::cout << std::string(LINE_WIDTH, '-') << std::endl;
	}

	inline void printDoubleLine()
	{
		std::cout << std::string(LINE_WIDTH, '=') << std::endl;
	}
}