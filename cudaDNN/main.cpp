#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>
#include <ostream>

#include "AI/AI.h"

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

	loadMnistFromBin("C:\\Users\\asahi\\Downloads\\mnist_data_train.bin", trainingData, sizeof(f32) * trainingData.size());
	loadMnistFromBin("C:\\Users\\asahi\\Downloads\\mnist_label_train.bin", trainingLabel, sizeof(f32) * trainingLabel.size());
	loadMnistFromBin("C:\\Users\\asahi\\Downloads\\mnist_data_test.bin", testData, sizeof(f32) * testData.size());
	loadMnistFromBin("C:\\Users\\asahi\\Downloads\\mnist_label_test.bin", testLabel, sizeof(f32) * testLabel.size());

	mnistNormalizer(trainingData, trainingDataNum);
	mnistNormalizer(testData, testDataNum);
}


#if 1
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


	//データ形状は自分の手で決める。
	DataShape inputTrainingDataShape = { 1, 1, 28 * 28 };
	DataShape inputCorrectDataShape = { 1, 1,  28 * 28 };
	//訓練データと教師データの形状についてAIに教える
	DataFormat4DeepLearning format(trainingDataNum, 100, inputTrainingDataShape, inputCorrectDataShape);

	//////////////////////////////////////////
	//AIの準備
	//////////////////////////////////////////


	AI Aira{};
	Aira.addLayer<layer::Affine>(50, 0.001f);
	Aira.addLayer<layer::BatchNorm2d>();
	Aira.addLayer<layer::ReLU>();
	Aira.addLayer<layer::Affine>(784, 0.001f);
	Aira.addLayer<layer::BatchNorm2d>();
	Aira.setOptimizer<optimizer::Sgd>(0.0001f);
	Aira.setLossFunction<lossFunction::L2Loss>();

	Aira.build(format);

	//////////////////////////////////////////
	//学習ループ
	//////////////////////////////////////////

	Aira.deepLearning(trainingData.data(), trainingData.data());

	return 0;
}

#else
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


	//データ形状は自分の手で決める。
	DataShape inputTrainingDataShape = {1, 1, 28 * 28 };
	DataShape inputCorrectDataShape = { 1, 1, 1 };
	//訓練データと教師データの形状についてAIに教える
	DataFormat4DeepLearning format(trainingDataNum, 100, inputTrainingDataShape, inputCorrectDataShape);

	//////////////////////////////////////////
	//AIの準備
	//////////////////////////////////////////


	AI Aira{};
	Aira.addLayer<layer::Affine>(50, 0.001f);
	Aira.addLayer<layer::ReLU>();
	Aira.addLayer<layer::Affine>(10, 0.001f);
	Aira.setOptimizer<optimizer::Sgd>(0.001f);
	Aira.setLossFunction<lossFunction::CrossEntropyWithSM>();
	//Aira.setLossFunction<lossFunction::L2Loss>();

	Aira.build(format);

	//////////////////////////////////////////
	//学習ループ
	//////////////////////////////////////////

	Aira.deepLearning(trainingData.data(), trainingLabel.data());

	return 0;
}
#endif