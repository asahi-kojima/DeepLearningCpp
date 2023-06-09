#pragma once
#include <iostream>
#include <string>

#if 1
#include <chrono>
#endif


#define LINE_WIDTH 100

namespace Aoba
{
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

	/*inline void loadMnistFromCsv(std::string filePath, std::vector<u32>& data, std::vector<u32>& label, u32 printInterval = 1000)
	{
		std::ifstream ifs(filePath);
		std::cout << "load start [" << filePath << "]" << std::endl;

		std::string line;
		u32 currentLineNum = 1;
		while (std::getline(ifs, line))
		{
			int num;
			std::string numString;
			std::stringstream stringStream{ line };
			u32 order = 0;
			while (std::getline(stringStream, numString, ','))
			{
				num = atoi(numString.c_str());
				if (order == 0)
				{
					label.push_back(num);
				}
				else
				{
					data.push_back(num);
				}
				order++;
			}
			if (currentLineNum % printInterval == 0)
			{
				std::cout << "Current Line : " << currentLineNum << std::endl;
			}
			currentLineNum++;
		}
		std::cout << "load finish [" << filePath << "]\n" << std::endl;
	}

	inline void printMNIST(u32* data)
	{
		for (int i = 0; i < 28; i++)
		{
			for (int j = 0; j < 28; j++)
			{
				std::cout << (data[i * 28 + j] > 0 ? 1 : 0);
			}
			std::cout << "\n";
		}
	}*/
}


/*
std::ofstream fout;
	{
		fout.open("C:\\Users\\asahi\\Downloads\\mnist_dataTrain.bin", std::ios::out | std::ios::binary);
		fout.write(reinterpret_cast<char*>(&(dataTrain[0])), sizeof(u32) * 784 * 60000);
		fout.close();
	}
	{
		fout.open("C:\\Users\\asahi\\Downloads\\mnist_labelTrain.bin", std::ios::out | std::ios::binary);
		fout.write(reinterpret_cast<char*>(&(labelTrain[0])), sizeof(u32) * 60000);
		fout.close();
	}
	{
		fout.open("C:\\Users\\asahi\\Downloads\\mnist_dataTest.bin", std::ios::out | std::ios::binary);
		fout.write(reinterpret_cast<char*>(&(dataTest[0])), sizeof(u32) * 784 * 10000);
		fout.close();
	}
	{
		fout.open("C:\\Users\\asahi\\Downloads\\mnist_labelTest.bin", std::ios::out | std::ios::binary);
		fout.write(reinterpret_cast<char*>(&(labelTest[0])), sizeof(u32)  * 10000);
		fout.close();
	}
*/