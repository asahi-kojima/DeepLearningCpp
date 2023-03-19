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

	void AI::build(InputDataShape& shape, std::unique_ptr<optimizer::BaseOptimizer>&& optimizer)
	{
		//層が最低でも一つあるかのチェック
		assert(mLayerList.size() > 0);

		//層のメモリを構成する上で必要になるパラメータの設定を行う。
		setupLayerInfo(shape);

		//各層内におけるメモリの確保
		allocLayerMemory();

		//オプティマイザーの登録
		mOptimizer = std::move(optimizer);
	}

	constDataMemory AI::forward(f32* inputDataAddress)
	{
		mInputData.address = reinterpret_cast<flowDataType*>(inputDataAddress);

		for (auto& layer : mLayerList)
		{
			layer->forward();
		}

		return *mForwardResult;
	}

	constDataMemory AI::backward(f32*)
	{
		return constDataMemory();
	}


	void AI::optimize()
	{
		for (auto& layer : mLayerList)
		{
			mOptimizer->optimize(layer);
		}
	}

#pragma endregion

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

		mInputData.size = shape.batchSize * shape.channel * shape.height * shape.width * sizeof(flowDataType);

		for (auto& layer : mLayerList)
		{
			layer->setupLayerInfo(&dataShape);
		}
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

		//学習時の各層が参照する前層のデータのアドレスを登録
		constDataMemory* pInputData = &mInputData;

		for (auto& layer : mLayerList)
		{
			pInputData = layer->setInputData(pInputData);
		}
		mForwardResult = pInputData;

		for (auto rit = mLayerList.rbegin(); rit != mLayerList.rend(); rit++)
		{
			pInputData = (*rit)->setDInputData(pInputData);
		}
	}
	



}