#pragma once

#include <vector>
#include <iostream>
#include <string>
#include <cuda_runtime.h>

#include "../typeinfo.h"
#include "AIDataStructure.h"
#include "AIMacro.h"


#define LINE_WIDTH 100


namespace Aoba
{
	inline std::vector<f32> cpuDataFromGpuData(DataArray& data)
	{
		std::vector<f32> v;
		v.resize(data.size);

		CHECK(cudaMemcpy(v.data(), data.address, data.size * sizeof(f32), cudaMemcpyDeviceToHost));
		return v;
		return std::move(v);
	}



	inline void printLine()
	{
		std::cout << std::string(LINE_WIDTH, '-') << std::endl;
	}

	inline void printDoubleLine()
	{
		std::cout << std::string(LINE_WIDTH, '=') << std::endl;
	}

	inline void informationFormat(std::string s)
	{
#if 0  
		std::cout << "  _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ " << std::endl;
		std::cout << " / / / / / / / / / / / / / / / / / / / / / / / / / / / / " << std::endl;
		std::string bar = " / / / / / / / / / / / / / / / / / / / / / / / / / / / / ";
		unsigned int sideLength = (bar.length() - s.length()) / 2;
		std::cout << std::string(sideLength, ' ') + s + std::string(sideLength, ' ') << std::endl;
		std::cout << "/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/_/" << std::endl;

#else	
		std::string topBar = "  ";
		std::string topBarParts = "_ ";
		for (int i = 0; i < LINE_WIDTH / 2 - 1; i++)
		{
			topBar = topBar + topBarParts;
		}
		std::cout << topBar << std::endl;

		std::string bar = "";
		std::string parts = " /";
		for (int i = 0; i < LINE_WIDTH / 2; i++)
		{
			bar = bar + parts;
		}
		std::cout << bar << std::endl;

		unsigned int sideLength = (bar.length() - s.length()) / 2;
		std::cout << std::string(sideLength, ' ') + s + std::string(sideLength, ' ') << std::endl;

		std::string underBar = "/";
		std::string underBarParts = "_/";
		for (int i = 0; i < LINE_WIDTH / 2 - 1; i++)
		{
			underBar = underBar + underBarParts;
		}
		std::cout << underBar << std::endl;
#endif
	}

	inline void printTitle(std::string s)
	{
		std::string topBar = "  ";
		std::string topBarParts = "_ ";
		for (int i = 0; i < LINE_WIDTH / 2 - 1; i++)
		{
			topBar = topBar + topBarParts;
		}
		std::cout << topBar << std::endl;

		std::string bar = "";
		std::string parts = " /";
		for (int i = 0; i < LINE_WIDTH / 2; i++)
		{
			bar = bar + parts;
		}
		std::cout << bar << std::endl;

		unsigned int sideLength = (bar.length() - s.length()) / 2;
		std::cout << std::string(sideLength, ' ') + s + std::string(sideLength, ' ') << std::endl;

		std::string underBar = "/";
		std::string underBarParts = "_/";
		for (int i = 0; i < LINE_WIDTH / 2 - 1; i++)
		{
			underBar = underBar + underBarParts;
		}
		std::cout << underBar << std::endl;
	}

	inline void printLayerName(std::string s)
	{
		std::cout << s << std::endl;
	}

	inline std::string addSpace2String(std::string s, u32 width = 15)
	{
		if (width <= s.length())
		{
			assert(0);
		}

		return s + std::string(width - s.length(), ' ');
	}

	inline void printProperty(std::string s, u32 value)
	{
		std::cout << std::string("	") + addSpace2String(s) + std::string(" = ") << value << std::endl;
	}

	inline void printProperty(std::string s, f32 value)
	{
		std::cout << std::string("	") + addSpace2String(s) + std::string(" = ") << value << std::endl;
	}

	inline void print2dProperty(std::string s, u32 value0, u32 value1)
	{
		std::cout << std::string("	") + addSpace2String(s) + std::string(" = (")  << value0 << ", " << value1 << ")" << std::endl;
	}

	inline void print3dProperty(std::string s, u32 value0, u32 value1, u32 value2)
	{
		std::cout << std::string("	") + addSpace2String(s) + std::string(" = (") << value0 << ", " << value1 << ", " << value2 << ")" << std::endl;
	}

	inline void print3dProperty(std::string s, DataShape shape)
	{
		std::cout << std::string("	") + addSpace2String(s) + std::string(" = (") << shape.channel << ", " << shape.height << ", " << shape.width << ")" << std::endl;
	}
}