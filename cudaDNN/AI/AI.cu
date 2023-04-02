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

	void AI::addLayer(std::unique_ptr<layer::BaseLayer>&& pLayer)
	{
		mLayerList.push_back(std::forward<std::unique_ptr<layer::BaseLayer>>(pLayer));
	}

	void AI::build(InputDataInterpretation& interpretation, std::unique_ptr<optimizer::BaseOptimizer>&& optimizer, std::unique_ptr<lossFunction::BaseLossFunction>&& lossFunction)
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
		mInterpretation = interpretation;
		setupLayerInfo(mInterpretation.shape);

		//各層内におけるメモリの確保
		allocLayerMemory();

	}


	void AI::deepLearning(f32* pTrainingData, f32* pTrainingLabel)
	{
		mInputTrainingDataStartAddressOnCpu = pTrainingData;
		mInputTrainingLableStartAddressOnCpu = pTrainingLabel;

		mInputTrainingData.size = mInterpretation.shape.batchSize * mInterpretation.elementNum;
		mInputTrainingData.byteSize = mInterpretation.shape.batchSize * mInterpretation.byteSize;
		mInputLabelData.size = mInterpretation.shape.batchSize * 1;
		mInputLabelData.byteSize = mInterpretation.shape.batchSize * sizeof(f32);

#if _DEBUG
		mInputTrainingDataForGpuDebug.size = mInterpretation.shape.batchSize * mInterpretation.elementNum;
		mInputTrainingDataForGpuDebug.byteSize = mInterpretation.shape.batchSize * mInterpretation.byteSize;
		mInputLabelDataForGpuDebug.size = mInterpretation.shape.batchSize * 1;
		mInputLabelDataForGpuDebug.byteSize = mInterpretation.shape.batchSize * sizeof(f32);
#endif

		//
		//順伝搬用のデータをここで準備する。
		//
		auto printer = [](std::string name, u32 value, u32 stringLen = 15)
		{
			u32 res = stringLen - name.length();
			std::cout << name + std::string(' ', res) << " = " << value << "\n";
		};
		std::cout << "TrainingData setup now" << std::endl;
		printer("TotalData num", mInterpretation.totalDataNum);
		printer("channel", mInterpretation.shape.channel);
		printer("height", mInterpretation.shape.height);
		printer("width", mInterpretation.shape.width);

		u32 loopTime = mInterpretation.totalDataNum / mInterpretation.shape.batchSize;
		for (u32 loop = 0; loop < loopTime; loop++)
		{
			u32 offset = (mInterpretation.shape.batchSize * mInterpretation.elementNum) * loop;
			if (mIsGpuAvailable)
			{
				mInputTrainingData.address = mInputTrainingDataStartAddressOnGpu + offset;
				mInputLabelData.address = mInputTrainingLableStartAddressOnGpu + offset;
#if _DEBUG
				mInputTrainingDataForGpuDebug.address = mInputTrainingDataStartAddressOnCpu + offset;
				mInputLabelDataForGpuDebug.address = mInputTrainingLableStartAddressOnCpu + offset;
#endif
			}
			else
			{
				mInputTrainingData.address = mInputTrainingDataStartAddressOnCpu + offset;
				mInputLabelData.address = mInputTrainingLableStartAddressOnCpu + offset;
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
		//GPU利用可能かの処理が入る。
		mIsGpuAvailable = true;


		for (auto& layer : mLayerList)
		{
			layer->setIsGpuAvailable(mIsGpuAvailable);
		}

		mLossFunction->setIsGpuAvailable(mIsGpuAvailable);

		mOptimizer->setIsGpuAvailable(mIsGpuAvailable);
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

		mInputTrainingData.size = shape.batchSize * shape.channel * shape.height * shape.width;
#if _DEBUG
		if (mIsGpuAvailable)
		{
			mInputTrainingDataForGpuDebug.size = shape.batchSize * shape.channel * shape.height * shape.width;
		}
#endif

		for (auto& layer : mLayerList)
		{
			layer->setupLayerInfo(&dataShape);
		}

		mLossFunction->setupDataShape(dataShape);
	}

	/// <summary>
	///　各層におけるパラメータのためのメモリ確保や初期化、
	/// そして学習時に各層が必要となる前の層の出力データのアドレスを登録。
	/// </summary>
	void AI::allocLayerMemory()
	{
		//GPU状のメモリの確保やそれの初期化
		for (auto& layer : mLayerList)
		{
			layer->initialize();
		}

		mLossFunction->initialize();

		//学習時の各層が参照する前層のデータのアドレスを登録
		//まず基点となるデータをセット
		DataMemory* pInputData = &mInputTrainingData;
#if _DEBUG
		DataMemory* pInputDataForGpuDebug = &mInputTrainingDataForGpuDebug;
#endif

		//ここで各層に参照するべきデータを順に渡していく。
		for (auto& layer : mLayerList)
		{
			layer->setInputData(pInputData);
#if _DEBUG
			if (mIsGpuAvailable)
			{
				layer->setInputDataForGpuDebug(pInputDataForGpuDebug);
			}
#endif
		}

		//損失関数に渡したり、その他用途のために順伝搬の出力をここにセットする。
		mForwardResult = pInputData;
#if _DEBUG
		if (mIsGpuAvailable)
		{
			mForwardResultForGpuDebug = pInputDataForGpuDebug;
		}
#endif

		//損失関数に順伝搬の結果を渡す。
		mLossFunction->setInput(mForwardResult, &mInputLabelData);
#if _DEBUG
		if (mIsGpuAvailable)
		{
			mLossFunction->setInputForGpuDebug(mForwardResultForGpuDebug, &mInputLabelDataForGpuDebug);
		}
#endif


		DataMemory* pDInputData = &(mLossFunction->mDInputData);
#if _DEBUG
		DataMemory* pDInputDataForGpuDebug = nullptr;
		if (mIsGpuAvailable)
		{
			pDInputDataForGpuDebug = &(mLossFunction->mDInputDataForGpuDebug);
		}
#endif
		for (auto rit = mLayerList.rbegin(); rit != mLayerList.rend(); rit++)
		{
			(*rit)->setDInputData(pDInputData);
#if _DEBUG
			if (mIsGpuAvailable)
			{
				(*rit)->setDInputDataForGpuDebug(pDInputDataForGpuDebug);
			}
#endif
		}
	}

	void AI::forward()
	{
		//順伝搬
		for (auto& layer : mLayerList)
		{
			layer->forward();
		}

		//損失の計算
		mLoss = mLossFunction->calcLossAndDInput();
	}

	void AI::backward()
	{
		for (auto riter = mLayerList.rbegin(), end = mLayerList.rend(); riter != end; riter++)
		{
			(*riter)->backward();
		}
	}


	void AI::optimize()
	{
		for (auto& layer : mLayerList)
		{
			mOptimizer->optimize(layer);
		}
	}

#pragma endregion
}