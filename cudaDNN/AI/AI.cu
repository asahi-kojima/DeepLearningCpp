#include <iostream>
#include <thread>
#include <random>
#include <cassert>

#include "./Layer/Layer.h"
#include "./Optimizer/Optimizer.h"
#include "AI.h"
#include "../commonGPU.cuh"
#include "../commonCPU.h"



namespace Aoba
{
#pragma region public

	AI::AI() = default;
	AI::~AI() = default;


	void AI::build(DataFormat4DeepLearning& format, std::unique_ptr<optimizer::BaseOptimizer>&& optimizer, std::unique_ptr<lossFunction::BaseLossFunction>&& lossFunction)
	{
		//層が最低でも一つあるかのチェック
		assert(mLayerList.size() > 0);

		//オプティマイザーの登録
		assert(optimizer != nullptr);
		mOptimizer = std::move(optimizer);

		//損失関数の登録
		assert(lossFunction != nullptr);
		mLossFunction = std::move(lossFunction);

		//GPUの利用が可能かチェックし、情報を出力。また全ての層にその情報を送る。
		checkGpuIsAvailable();

		//層のメモリを構成する上で必要になるパラメータの設定を行う。
		mDataFormat4DeepLearning = format;
		setupLayerInfo(mDataFormat4DeepLearning.trainingDataShape);

		//各層内におけるメモリの確保
		allocLayerMemory();

	}


	void AI::deepLearning(f32* pTrainingData, f32* pTrainingLabel, u32 epochs, f32 learningRate)
	{
		mOptimizer->setLearningRate(learningRate);

		dataSetup(pTrainingData, pTrainingLabel);

		//
		//順伝搬用のデータをここで準備する。
		//
		auto printer = [](std::string name, u32 value, u32 stringLen = 15)
		{
			u32 res = stringLen - name.length();
			std::string space = std::string(res, ' ');
			std::cout << name << space << " = " << value << "\n";
		};
		std::cout << "TrainingData setup now" << std::endl;
		printer("TotalData num", mDataFormat4DeepLearning.dataNum);
		printer("channel", mDataFormat4DeepLearning.trainingDataShape.channel);
		printer("height", mDataFormat4DeepLearning.trainingDataShape.height);
		printer("width", mDataFormat4DeepLearning.trainingDataShape.width);

		u32 loopTime = mDataFormat4DeepLearning.dataNum / mDataFormat4DeepLearning.trainingDataShape.batchSize;
		u32 batch = mDataFormat4DeepLearning.trainingDataShape.batchSize;
		auto progressBar = [](u32 currentLoop, u32 totalLoop, u32 length = 20)
		{
			u32 grid = totalLoop / length;
			std::string s = "\r";
			for (u32 i = 0; i < static_cast<u32>((static_cast<f32>(length) * currentLoop) / totalLoop); i++)
			{
				s += "=";
			}
			s += ">";
			int spaceLength = static_cast<s32>(length - s.length() + 1);
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
				progressBar(loop+1, loopTime);
				u32 offsetForTrainingData = (batch * mDataFormat4DeepLearning.eachTrainingDataSize) * loop;
				u32 offsetForCorrectLabel = (batch * mDataFormat4DeepLearning.eachCorrectDataSize) * loop;
				if (mIsGpuAvailable)
				{
					mInputTrainingDataOnGPU.address = mInputTrainingDataStartAddressOnGPU + offsetForTrainingData;
					mInputLabelDataOnGPU.address = mInputTrainingLableStartAddressOnGPU + offsetForCorrectLabel;
				}
				else
				{
					mInputTrainingDataOnCPU.address = mInputTrainingDataStartAddressOnCPU + offsetForTrainingData;
					mInputLabelDataOnCPU.address = mInputTrainingLableStartAddressOnCPU + offsetForCorrectLabel;
				}
				forward();
#if _DEBUG
				//
				//ここで整合性チェック
				//
#endif



				backward();
#if _DEBUG
				//
				//ここで整合性チェック
				//
#endif


				optimize();
#if _DEBUG
				//
				//ここで整合性チェック
				//
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
		mIsGpuAvailable = true;
	}

	/// <summary>
	/// 各層の内部パラメータを計算する。
	/// flowDataShapeには入力データの形状が入っているので、
	/// それを基にカーネルのサイズやパラメータの数を計算。
	/// </summary>
	void AI::setupLayerInfo(InputDataShape& shape)
	{
		InputDataShape dataShape;
		{
			dataShape.batchSize = shape.batchSize;
			dataShape.channel = shape.channel;
			dataShape.height = shape.height;
			dataShape.width = shape.width;
		}


		for (auto& layer : mLayerList)
		{
			layer->setupLayerInfo(&dataShape);
		}

		mLossFunction->setupDataShape(dataShape);

		if (mIsGpuAvailable)
		{
			mInputTrainingDataOnGPU.size = shape.batchSize * shape.channel * shape.height * shape.width;
		}
		else
		{
			mInputTrainingDataOnCPU.size = shape.batchSize * shape.channel * shape.height * shape.width;
		}
	}

	/// <summary>
	///　各層におけるパラメータのためのメモリ確保や初期化、
	/// そして学習時に各層が必要となる前の層の出力データのアドレスを登録。
	/// </summary>
	void AI::allocLayerMemory()
	{
		if (mIsGpuAvailable)
		{
			//GPU状のメモリの確保やそれの初期化
			for (auto& layer : mLayerList)
			{
				layer->initializeOnGPU();
			}

			mLossFunction->initializeOnGPU();

			//学習時の各層が参照する前層のデータのアドレスを登録
			//まず基点となるデータをセット
			DataMemory* pInputDataOnGPU = &mInputTrainingDataOnGPU;

			//ここで各層に参照するべきデータを順に渡していく。
			for (auto& layer : mLayerList)
			{
				layer->setInputDataOnGPU(pInputDataOnGPU);
			}

			//損失関数に渡したり、その他用途のために順伝搬の出力をここにセットする。
			mForwardResultOnGPU = pInputDataOnGPU;

			//損失関数に順伝搬の結果を渡す。
			mLossFunction->setInputOnGPU(mForwardResultOnGPU, &mInputLabelDataOnGPU);


			DataMemory* pDInputDataOnGPU = mLossFunction->getDInputDataOnGPU();

			for (auto rit = mLayerList.rbegin(); rit != mLayerList.rend(); rit++)
			{
				(*rit)->setDInputDataOnGPU(pDInputDataOnGPU);
			}
		}
		else
		{
			//GPU状のメモリの確保やそれの初期化
			for (auto& layer : mLayerList)
			{
				layer->initializeOnCPU();
			}

			mLossFunction->initializeOnCPU();

			//学習時の各層が参照する前層のデータのアドレスを登録
			//まず基点となるデータをセット
			DataMemory* pInputDataOnCPU = &mInputTrainingDataOnCPU;

			//ここで各層に参照するべきデータを順に渡していく。
			for (auto& layer : mLayerList)
			{
				layer->setInputDataOnCPU(pInputDataOnCPU);
			}

			//損失関数に渡したり、その他用途のために順伝搬の出力をここにセットする。
			mForwardResultOnCPU = pInputDataOnCPU;

			//損失関数に順伝搬の結果を渡す。
			mLossFunction->setInputOnCPU(mForwardResultOnCPU, &mInputLabelDataOnCPU);


			DataMemory* pDInputDataOnCPU = mLossFunction->getDInputDataOnCPU();

			for (auto rit = mLayerList.rbegin(); rit != mLayerList.rend(); rit++)
			{
				(*rit)->setDInputDataOnCPU(pDInputDataOnCPU);
			}
		}
	}

	void AI::dataSetup(f32* pTrainingData, f32* pTrainingLabel)
	{
		mInputTrainingDataStartAddressOnCPU = pTrainingData;
		mInputTrainingLableStartAddressOnCPU = pTrainingLabel;

		u32 batch = mDataFormat4DeepLearning.trainingDataShape.batchSize;
		if (mIsGpuAvailable)
		{
			mInputTrainingDataOnGPU.size = batch * mDataFormat4DeepLearning.eachTrainingDataSize;
			mInputTrainingDataOnGPU.byteSize = mInputTrainingDataOnGPU.size * sizeof(f32);
			mInputLabelDataOnGPU.size = batch * mDataFormat4DeepLearning.eachCorrectDataSize;
			mInputLabelDataOnGPU.byteSize = mInputLabelDataOnGPU.size * sizeof(f32);

			CHECK(cudaMalloc((void**)(&mInputTrainingDataStartAddressOnGPU), mDataFormat4DeepLearning.dataNum * mDataFormat4DeepLearning.eachTrainingDataSize * sizeof(f32)));
			CHECK(cudaMalloc((void**)(&mInputTrainingLableStartAddressOnGPU), mDataFormat4DeepLearning.dataNum * mDataFormat4DeepLearning.eachCorrectDataSize * sizeof(f32)));

			CHECK(cudaMemcpy(mInputTrainingDataStartAddressOnGPU, mInputTrainingDataStartAddressOnCPU, mDataFormat4DeepLearning.dataNum * mDataFormat4DeepLearning.eachTrainingDataSize * sizeof(f32), cudaMemcpyHostToDevice));
			CHECK(cudaMemcpy(mInputTrainingLableStartAddressOnGPU, mInputTrainingLableStartAddressOnCPU, mDataFormat4DeepLearning.dataNum * mDataFormat4DeepLearning.eachCorrectDataSize * sizeof(f32), cudaMemcpyHostToDevice));
		}
		else
		{
			mInputTrainingDataOnCPU.size = batch * mDataFormat4DeepLearning.eachTrainingDataSize;
			mInputTrainingDataOnCPU.byteSize = mInputTrainingDataOnCPU.size * sizeof(f32);
			mInputLabelDataOnCPU.size = batch * mDataFormat4DeepLearning.eachCorrectDataSize;
			mInputLabelDataOnCPU.byteSize = mInputLabelDataOnCPU.size * sizeof(f32);
		}
	}

	void AI::forward()
	{
		if (mIsGpuAvailable)
		{
			//順伝搬
			for (auto& layer : mLayerList)
			{
				layer->forwardOnGPU();
			}

			//損失の計算
			mLossOnGPU = mLossFunction->calcLossAndDInputOnGPU();
		}
		else
		{
			//順伝搬
			for (auto& layer : mLayerList)
			{
				layer->forwardOnCPU();
			}

			//損失の計算
			mLossOnCPU = mLossFunction->calcLossAndDInputOnCPU();
		}
	}

	void AI::backward()
	{
		if (mIsGpuAvailable)
		{
			for (auto riter = mLayerList.rbegin(), end = mLayerList.rend(); riter != end; riter++)
			{
				(*riter)->backwardOnGPU();
			}
		}
		else
		{
			for (auto riter = mLayerList.rbegin(), end = mLayerList.rend(); riter != end; riter++)
			{
				(*riter)->backwardOnCPU();
			}
		}
	}


	void AI::optimize()
	{
		if (mIsGpuAvailable)
		{
			for (auto& layer : mLayerList)
			{
				mOptimizer->optimizeOnGPU(layer);
			}
		}
		else
		{
			for (auto& layer : mLayerList)
			{
				mOptimizer->optimizeOnCPU(layer);
			}
		}

	}

#pragma endregion
}