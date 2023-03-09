#include "Machine.h"


int main()
{
	u32 exitCode = miduho::Machine::getInstance().entry();
	return exitCode;
}