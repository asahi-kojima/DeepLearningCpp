#include <iostream>
#include <thread>
#include <random>
#include <cassert>

#include "./Layer/Layer.h"
#include "./Optimizer/Optimizer.h"
#include "AI.h"
#include "AIMacro.h"
#include "../commonGPU.cuh"
#include "../commonCPU.h"



namespace Aoba
{
#pragma region public

	AI::AI() = default;
	AI::~AI() = default;


	void AI::build(DataFormat4DeepLearning& format)
	{
		//------------------------------------------------------------------
		//層が最低でも一つあるかのチェック
		//------------------------------------------------------------------
		assert(mLayerList.size() > 0);

		//------------------------------------------------------------------
		//オプティマイザーの登録
		//------------------------------------------------------------------
		if (mOptimizer == nullptr)
		{
			std::cout << "Optimizer is not defined. Default Optimizer is set..." << std::endl;
			mOptimizer = std::make_unique<optimizer::Sgd>(0.001f);
		}

		//------------------------------------------------------------------
		//損失関数の登録
		//------------------------------------------------------------------
		if (mLossFunction == nullptr)
		{
			std::cout << "Loss function is not defined. Default Loss function is set..." << std::endl;
			mLossFunction = std::make_unique<lossFunction::CrossEntropyWithSM>();
		}

		//------------------------------------------------------------------
		//GPUの利用が可能かチェックし、情報を出力。また全ての層にその情報を送る。
		//------------------------------------------------------------------
		checkGpuIsAvailable();

		//------------------------------------------------------------------
		//データフォーマットを保存
		//------------------------------------------------------------------
		mDataFormat4DeepLearning = format;

		//------------------------------------------------------------------
		// 各層における入力データの形状登録
		// 及び、層のメモリを構成する上で必要になるパラメータの設定を行う。
		//------------------------------------------------------------------
		initializeLayer();
	}


	void AI::deepLearning(f32* pTrainingData, f32* pCorrectData, u32 epochs)
	{
		//mOptimizer->setLearningRate(learningRate);


		//------------------------------------------------------------------
		//層の情報を表示
		//------------------------------------------------------------------
		std::cout << "[ Layer Information ]" << std::endl;
		for (auto& layer : mLayerList)
		{
			layer->printLayerInfo();
		}
		std::cout << std::endl;
		
		
		//------------------------------------------------------------------
		//訓練データの情報を表示
		//------------------------------------------------------------------
		std::cout << "[ Training Data Information ]" << std::endl;
#pragma region print_TrainingData_infomation
		auto printer = [](std::string name, u32 value, u32 stringLen = 15)
		{
			u32 res = stringLen - name.length();
			std::string space = std::string(res, ' ');
			std::cout << name << space << " = " << value << "\n";
		};
		printer("TotalData num", mDataFormat4DeepLearning.dataNum);
		printer("channel", mDataFormat4DeepLearning.trainingDataShape.channel);
		printer("height", mDataFormat4DeepLearning.trainingDataShape.height);
		printer("width", mDataFormat4DeepLearning.trainingDataShape.width);
		std::cout << std::endl;
#pragma endregion


		//------------------------------------------------------------------
		//順伝搬用のデータをここで準備する。
		//------------------------------------------------------------------
		dataSetup(pTrainingData, pCorrectData);


		//------------------------------------------------------------------
		//深層学習の実行箇所
		//------------------------------------------------------------------
		std::cout << "[ Deep Learning Start ]" << std::endl;
		u32 loopTime = mDataFormat4DeepLearning.dataNum / mDataFormat4DeepLearning.batchSize;
		u32 batch = mDataFormat4DeepLearning.batchSize;
		auto progressBar = [](u32 currentLoop, u32 totalLoop, u32 length = 20)
		{
			u32 grid = totalLoop / length;
			std::string s = "\r";
			for (u32 i = 0; i < static_cast<u32>((static_cast<f32>(length) * currentLoop) / totalLoop); i++)
			{
				s += "=";
			}
			s += ">";
			s32 spaceLength = static_cast<s32>(length - s.length() + 2);
			for (s32 i = 0; i < spaceLength; i++)
			{
				s += " ";
			}
			s += " " + std::to_string(static_cast<u32>(static_cast<f32>(currentLoop * 100) / totalLoop)) + "/100";
			printf(s.c_str());
		};

		for (u32 epoch = 0; epoch < epochs; epoch++)
		{
			std::cout << "epoch = " << epoch + 1 << std::endl;
			f32 loss = 0.0f;
			std::cout << "deep learning now" << std::endl;
			for (u32 loop = 0; loop < loopTime; loop++)
			{
				progressBar(loop + 1, loopTime);
				u32 offsetForTrainingData = (batch * mDataFormat4DeepLearning.eachTrainingDataSize) * loop;
				u32 offsetForCorrectData = (batch * mDataFormat4DeepLearning.eachCorrectDataSize) * loop;
				if (mIsGpuAvailable)
				{
					mInputTrainingDataOnGPU.address = mInputTrainingDataStartAddressOnGPU + offsetForTrainingData;
					mInputCorrectDataOnGPU.address = mInputCorrectDataStartAddressOnGPU + offsetForCorrectData;
				}
				else
				{
					mInputTrainingDataOnCPU.address = mInputTrainingDataStartAddressOnCPU + offsetForTrainingData;
					mInputCorrectDataOnCPU.address = mInputCorrectDataStartAddressOnCPU + offsetForCorrectData;
				}


				forward();
#if _DEBUG
				//------------------------------------------------------------------
				//ここで整合性チェック
				//------------------------------------------------------------------
#endif



				backward();
#if _DEBUG
				//------------------------------------------------------------------
				//ここで整合性チェック
				//------------------------------------------------------------------
#endif


				optimize();
#if _DEBUG
				//------------------------------------------------------------------
				//ここで整合性チェック
				//------------------------------------------------------------------
#endif
				if (mIsGpuAvailable)
					loss += mLossOnGPU;
				else
					loss += mLossOnCPU;
			}
			std::cout << "\n";
			std::cout << "current loss = " << loss / loopTime << "\n" << std::endl;
		}
	}


	DataMemory AI::operator()(f32* inputData)
	{
		return DataMemory();
	}

#pragma endregion

#pragma region private
	void AI::checkGpuIsAvailable()
	{
		std::cout << "[ GPU Information ]" << std::endl;
		mIsGpuAvailable = true;
		std::cout <<  std::endl;
	}


	/// <summary>
	///　各層におけるパラメータのためのメモリ確保や初期化、
	/// そして学習時に各層が必要となる前の層の出力データのアドレスを登録。
	/// </summary>
	void AI::initializeLayer()
	{
		if (mIsGpuAvailable)
		{
			INITIALIZE_ON_(GPU);
		}
		else
		{
			INITIALIZE_ON_(CPU);
		}
	}

	void AI::dataSetup(f32* pTrainingData, f32* pCorrectData)
	{
		mInputTrainingDataStartAddressOnCPU = pTrainingData;
		mInputCorrectDataStartAddressOnCPU = pCorrectData;

		u32 batch = mDataFormat4DeepLearning.batchSize;
		if (mIsGpuAvailable)
		{
			mInputTrainingDataOnGPU.size = batch * mDataFormat4DeepLearning.eachTrainingDataSize;
			mInputTrainingDataOnGPU.byteSize = mInputTrainingDataOnGPU.size * sizeof(f32);
			mInputCorrectDataOnGPU.size = batch * mDataFormat4DeepLearning.eachCorrectDataSize;
			mInputCorrectDataOnGPU.byteSize = mInputCorrectDataOnGPU.size * sizeof(f32);

			CHECK(cudaMalloc((void**)(&mInputTrainingDataStartAddressOnGPU), mDataFormat4DeepLearning.dataNum * mDataFormat4DeepLearning.eachTrainingDataSize * sizeof(f32)));
			CHECK(cudaMalloc((void**)(&mInputCorrectDataStartAddressOnGPU), mDataFormat4DeepLearning.dataNum * mDataFormat4DeepLearning.eachCorrectDataSize * sizeof(f32)));

			CHECK(cudaMemcpy(mInputTrainingDataStartAddressOnGPU, mInputTrainingDataStartAddressOnCPU, mDataFormat4DeepLearning.dataNum * mDataFormat4DeepLearning.eachTrainingDataSize * sizeof(f32), cudaMemcpyHostToDevice));
			CHECK(cudaMemcpy(mInputCorrectDataStartAddressOnGPU, mInputCorrectDataStartAddressOnCPU, mDataFormat4DeepLearning.dataNum * mDataFormat4DeepLearning.eachCorrectDataSize * sizeof(f32), cudaMemcpyHostToDevice));
		}
		else
		{
			mInputTrainingDataOnCPU.size = batch * mDataFormat4DeepLearning.eachTrainingDataSize;
			mInputTrainingDataOnCPU.byteSize = mInputTrainingDataOnCPU.size * sizeof(f32);
			mInputCorrectDataOnCPU.size = batch * mDataFormat4DeepLearning.eachCorrectDataSize;
			mInputCorrectDataOnCPU.byteSize = mInputCorrectDataOnCPU.size * sizeof(f32);
		}
	}

	void AI::forward()
	{
		if (mIsGpuAvailable)
		{
			FORWARD_ON_(GPU);
		}
		else
		{
			FORWARD_ON_(CPU);
		}
	}

	void AI::backward()
	{
		if (mIsGpuAvailable)
		{
			BACKWARD_ON_(GPU);
		}
		else
		{
			BACKWARD_ON_(CPU);
		}
	}


	void AI::optimize()
	{
		if (mIsGpuAvailable)
		{
			OPTIMIZE_ON_(GPU);
		}
		else
		{
			OPTIMIZE_ON_(CPU);
		}
	}

#pragma endregion
}