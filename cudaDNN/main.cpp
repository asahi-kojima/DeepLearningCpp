#include "Machine.h"
#include <iostream>


int main()
{
	std::cout << "Deep Learning Application Start" << std::endl;
	u32 exitCode = miduho::Machine::getInstance().entry();
	return exitCode;
}