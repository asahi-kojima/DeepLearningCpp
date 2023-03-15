#include <iostream>
#include <thread>
#include <random>

#include "./Layer/Layer.h"
#include "./Optimizer/Optimizer.h"
#include "Machine.h"
#include "commonGPU.cuh"
#include "commonCPU.h"

#define CREATELAYER(classname, ...) std::make_unique<classname>(__VA_ARGS__)


namespace miduho
{
	Machine::Machine() = default;
	Machine::~Machine() = default;

	u32 Machine::entry()
	{
#ifdef GPU_AVAILABLE
		printLine();
		std::cout << "GPU Information" << std::endl;
		printGpuDriverInfo();
		printLine();
#endif // GPUA_VAILABLE


		bool isInitializeSuccess = initialize();
		if (!isInitializeSuccess)
		{
			return 1;
		}

		bool isPreProcessSuccess = preProcess();
		if (!isPreProcessSuccess)
		{
			return 1;
		}

		bool isMainProcessSuccess = mainProcess();
		if (!isMainProcessSuccess)
		{
			return 1;
		}

		bool isPostProcessSuccess = postProcess();
		if (!isPostProcessSuccess)
		{
			return 1;
		}

		bool isTerminateSuccess = terminate();
		if (!isTerminateSuccess)
		{
			return 1;
		}

		return 0;
	}

	////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////
	bool Machine::initialize()
	{
		printDoubleLine();
		std::cout << "Machine initialize start\n" << std::endl;
		{
			//層構造の決定
			entryLayer(CREATELAYER(layer::Affine, 10));
			entryLayer(CREATELAYER(layer::ReLU));
			entryLayer(CREATELAYER(layer::Affine, 20));
			entryLayer(CREATELAYER(layer::ReLU));
			entryLayer(CREATELAYER(layer::Affine, 30));
			entryLayer(CREATELAYER(layer::ReLU));
			entryLayer(CREATELAYER(layer::Affine, 40));
			entryLayer(CREATELAYER(layer::ReLU));
			entryLayer(CREATELAYER(layer::Affine, 50));

			//層のメモリを構成する上で必要になるパラメータの設定を行う。
			setupLayerInfo();

			//メモリの確保
			initializeLayer();

			//オプティマイザー
			mOptimizer = std::make_unique<optimizer::Adam>();
		}
		std::cout << "Machine initialize finish" << std::endl;
		printDoubleLine();
		return true;
	}

	void Machine::entryLayer(std::unique_ptr<layer::BaseLayer>&& pLayer)
	{
		mLayerList.push_back(std::forward<std::unique_ptr<layer::BaseLayer>>(pLayer));
	}

	/// <summary>
	/// 各層の内部パラメータを計算する。
	/// flowDataShapeには入力データの形状が入っているので、
	/// それを基にカーネルのサイズやパラメータの数を計算。
	/// </summary>
	void Machine::setupLayerInfo()
	{
		layer::BaseLayer::FlowDataFormat flowDataShape;
		{
			flowDataShape.batchSize = mLearningData.dataShape.batchSize;
			flowDataShape.channel = mLearningData.dataShape.channel;
			flowDataShape.height = mLearningData.dataShape.height;
			flowDataShape.width = mLearningData.dataShape.width;
		}
		for (auto& layer : mLayerList)
		{
			layer->setupLayerInfo(&flowDataShape);
		}
	}

	/// <summary>
	///　各層におけるパラメータのためのメモリ確保や初期化、
	/// そして学習時に各層が必要となる前の層の出力データのアドレスを登録。
	/// </summary>
	void Machine::initializeLayer()
	{
		//GPU状のメモリの確保やそれの初期化
		for (auto& layer : mLayerList)
		{
			layer->initialize();
		}

		//学習時の各層が参照する前層のデータのアドレスを登録
		dataMemory* pInputData = &mLearningData.InputData;

		for (auto& layer : mLayerList)
		{
			layer->setInputData(pInputData);
			pInputData = layer->getDataMemory();
		}
		for (auto rit = mLayerList.rbegin(); rit != mLayerList.rend(); rit++)
		{
			(*rit)->setDInputData(pInputData);
			pInputData = (*rit)->getDDataMemory();
		}


	}

	////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////
	bool Machine::preProcess()
	{
		printDoubleLine();
		std::cout << "Machine preProcess start" << std::endl;
		{
			makeTestData();

		}
		std::cout << "Machine preProcess finish" << std::endl;
		printDoubleLine();
		return true;
	}


	////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////
	bool Machine::mainProcess()
	{
		printDoubleLine();
		std::cout << "Machine mainProcess start" << std::endl;
		{
			startLearning();
		}
		std::cout << "Machine mainProcess finish" << std::endl;
		printDoubleLine();
		return true;
	}

	void Machine::startLearning()
	{
		u32 totalEpoch = 100;
		for (u32 epoch = 0; epoch < totalEpoch; epoch++)
		{
			std::cout << "Epoch " << epoch << "\n";

			//バッチによる分割数を決定
			u32 totalBatchTime = 1000 / mLearningData.dataShape.batchSize;
			for (u32 batchTime = 0; batchTime < totalBatchTime; batchTime++)
			{
				//順伝搬
				for (auto& layer : mLayerList)
				{
					layer->forward();
				}

				//逆伝搬
				for (auto rit = mLayerList.rbegin(); rit != mLayerList.rend(); rit++)
				{
					(*rit)->backward();
				}

				//パラメータの更新
				for (auto& layer : mLayerList)
				{
					mOptimizer->optimize(*layer);
				}
			}
		}
	}


	////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////
	bool Machine::postProcess()
	{
		printDoubleLine();
		std::cout << "Machine postProcess start" << std::endl;
		{
			

		}
		std::cout << "Machine postProcess finish" << std::endl;
		printDoubleLine();
		return true;
	}

	////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////////////////////////
	bool Machine::terminate()
	{
		printDoubleLine();
		std::cout << "Machine terminate start" << std::endl;
		{



		}
		std::cout << "Machine terminate finish" << std::endl;
		printDoubleLine();
		return true;
	}


	void Machine::makeTestData()
	{
		mLearningData.InputData.size = (mLearningData.dataShape.batchSize * mLearningData.dataShape.width);
		flowDataType* tmp = new flowDataType[mLearningData.InputData.size];
		for (u32 idx = 0; idx < mLearningData.InputData.size; idx++)
		{
			tmp[idx] = 0.1f;
		}

		CHECK(cudaMalloc((void**)(&(mLearningData.InputData.address)), mLearningData.InputData.size * sizeof(f32)));
		CHECK(cudaMemcpy(mLearningData.InputData.address, tmp, mLearningData.InputData.size * sizeof(f32), cudaMemcpyHostToDevice));

		delete[] tmp;



		/*mLearningData.allInputData.size = 50 * mLearningData.dataShape.batchSize * mLearningData.dataShape.width;
		CHECK(cudaMalloc((void**)(&(mLearningData.allInputData.address)), mLearningData.allInputData.size * sizeof(f32)));
		mLearningData.allTrainingData.size = 50 * mLearningData.dataShape.batchSize;
		CHECK(cudaMalloc((void**)(&(mLearningData.allTrainingData.address)), mLearningData.allTrainingData.size * sizeof(u32)));


		std::random_device device;
		std::default_random_engine engine(device());
		std::normal_distribution<float> dist(0.0f, 1.0);
		flowDataType* tmp = new flowDataType[mLearningData.allInputData.size];
		{
			for (u32 i = 0; i < 50 * mLearningData.dataShape.batchSize; i++)
			{
				f32 which = dist(engine);
				if (which > 0)
				{
					 
					mLearningData.allTrainingData.address[i] = 0;
				}
				else
				{
					mLearningData.allTrainingData.address[i] = 1;
				}
			}
		}
		delete[] tmp;

		mLearningData.InputData.size = mLearningData.dataShape.batchSize * mLearningData.dataShape.width;
		mLearningData.InputData.address = mLearningData.allInputData.address;
		mLearningData.inputDataOffset = 0;
		mLearningData.TrainingData.address = mLearningData.allInputData.address;
		mLearningData.trainingDataOffset = 0;*/
	}

}