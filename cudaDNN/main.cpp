#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <ostream>

#include "AI/AI.h"
#include "AI/Layer/Layer.h"
#include "AI/Optimizer/Optimizer.h"
#include "AI/LossFunction/LossFunction.h"
#include "commonGPU.cuh"


void loadMnistFromBin(std::string filePath, std::vector<f32>& data, u32 loadByteSize)
{
	std::cout << "load start [" << filePath << "]";

	std::ifstream fin(filePath, std::ios::in | std::ios::binary);
	if (!fin)
	{
		std::cout << "\nthis program can't open the file : " << filePath << "\n" << std::endl;
		return;
	}
	fin.read(reinterpret_cast<char*>(data.data()), loadByteSize);

	std::cout << "-----> load finish" << std::endl;
}

void printMNIST(f32* data)
{
	for (int i = 0; i < 28; i++)
	{
		for (int j = 0; j < 28; j++)
		{
			std::cout << (data[i * 28 + j] > 0.1 ? 1 : 0);
		}
		std::cout << "\n";
	}
	std::cout << "\n\n";
}

void mnistNormalizer(std::vector<f32>& v, u32 size)
{
	for (u32 i = 0; i < size; i++)
	{
		u32 offset = i * 784;
		
		f32 mu = 0;
		for (u32 j = 0; j < 784; j++)
		{
			mu += v[offset + j] / 784.0f;
		}

		f32 sigma2 = 0.0f;
		for (u32 j = 0; j < 784; j++)
		{
			sigma2 += (v[offset + j] - mu) * (v[offset + j] - mu) / 784.0f;
		}

		f32 sigma = std::sqrtf(sigma2);
		for (u32 j = 0; j < 784; j++)
		{
			v[offset + j] = (v[offset + j] - mu) / sigma;
		}
	}
}

void setupMnistData(std::vector<f32>& trainingData, std::vector<f32>& trainingLabel, std::vector<f32>& testData, std::vector<f32>& testLabel)
{
	constexpr u32 dataSize = 784;
	constexpr u32 trainingDataNum = 60000;
	constexpr u32 testDataNum = 10000;
	trainingData.resize(trainingDataNum * dataSize);
	trainingLabel.resize(trainingDataNum);
	testData.resize(testDataNum * dataSize);
	testLabel.resize(testDataNum);

	loadMnistFromBin("C:\\Users\\asahi\\Downloads\\mnist_data_train.bin",  trainingData, sizeof(f32) * trainingData.size());
	loadMnistFromBin("C:\\Users\\asahi\\Downloads\\mnist_label_train.bin", trainingLabel, sizeof(f32) * trainingLabel.size());
	loadMnistFromBin("C:\\Users\\asahi\\Downloads\\mnist_data_test.bin",   testData, sizeof(f32) * testData.size());
	loadMnistFromBin("C:\\Users\\asahi\\Downloads\\mnist_label_test.bin",  testLabel, sizeof(f32) * testLabel.size());

	printMNIST(testData.data() + 784 * 1000);
	printMNIST(trainingData.data() + 784 * 59999);
	mnistNormalizer(trainingData, trainingDataNum);
	mnistNormalizer(testData, testDataNum);
}

void setupMnistGpuData(Aoba::DataMemory& dataGPU, std::vector<f32>& data)
{
	dataGPU.size = data.size();
	CHECK(cudaMalloc((void**)(&(dataGPU.address)), data.size() * sizeof(Aoba::flowDataType)));
	CHECK(cudaMemcpy(dataGPU.address, data.data(), data.size() * sizeof(Aoba::flowDataType), cudaMemcpyHostToDevice));
#ifdef _DEBUG
	{
		std::vector<f32> tester(data.size());
		CHECK(cudaMemcpy(tester.data(), dataGPU.address, tester.size() * sizeof(Aoba::flowDataType), cudaMemcpyDeviceToHost));
	}
#endif // _DEBUG
}

int main()
{
	using namespace Aoba;
	//////////////////////////////////////////
	//データの準備
	//////////////////////////////////////////
	constexpr u32 dataSize = 784;
	constexpr u32 trainingDataNum = 60000;
	constexpr u32 testDataNum = 10000;
	std::vector<f32> trainingData, trainingLabel, testData, testLabel;
	setupMnistData(trainingData, trainingLabel, testData, testLabel);

	DataMemory trainingDataGPU, trainingLabelGPU, testDataGPU, testLabelGPU;
	setupMnistGpuData(trainingDataGPU, trainingData);
	setupMnistGpuData(trainingLabelGPU, trainingLabel);
	setupMnistGpuData(testDataGPU,testData);
	setupMnistGpuData(testLabelGPU,testLabel);



	//データ形状は自分の手で決める。
	AI::InputDataShape inputDataShape = { 100, 1, 1, 28 * 28 };

	//////////////////////////////////////////
	//AIの準備
	//////////////////////////////////////////


	AI Aira{};
	Aira.addLayer(CREATELAYER(layer::Affine, 300, 0.001f));
	Aira.addLayer(CREATELAYER(layer::ReLU));
	Aira.addLayer(CREATELAYER(layer::Affine, 60, 0.001f));
	Aira.addLayer(CREATELAYER(layer::ReLU));
	Aira.addLayer(CREATELAYER(layer::Affine, 10,0.001f));

	AI::InputData inputData(trainingData.data(), 28 * 28, trainingDataNum);

	Aira.build(inputDataShape, std::make_unique<optimizer::Sgd>(0.001f), std::make_unique<lossFunction::CrossEntropyWithSM>());

	//////////////////////////////////////////
	//学習ループ
	//////////////////////////////////////////

	for (u32 epoch = 0; epoch < 100; epoch++)
	{
		std::cout << "Epoch " << epoch << "start\n";
		f32 loss = 0.0f;
		f32 count = 0;
		for (u32 batchLoop = 0, end = trainingDataNum / inputDataShape.batchSize; batchLoop < end; batchLoop++)
		{
			f32* dataAddress = trainingDataGPU.address + batchLoop * (dataSize * inputDataShape.batchSize);
			f32* labelAddress = trainingLabel.data() + batchLoop * (inputDataShape.batchSize);


			Aira.learning(dataAddress, labelAddress);
			f32 loss = Aira.getLoss();
		}

		std::cout << "average loss = " << loss / count << std::endl;
	}
	
	return 0;
}