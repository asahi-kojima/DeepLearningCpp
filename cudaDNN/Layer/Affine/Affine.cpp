#include <random>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

#include "Affine.h"
#include "commonCPU.h"

namespace miduho::layer
{


	Affine::Affine(u32 outputSize)
		:mBatchSize(0)
		,mInputSize(0)
		,mOutputSize(outputSize)
	{
	}


	Affine::~Affine()
	{
	}

	void Affine::initialize(flowDataFormat* pInputData)
	{
#if _DEBUG
		printLine();
		std::cout << "Affine initialize start" << std::endl;
#endif
		mBatchSize = pInputData->batchSize;
		mInputSize = pInputData->width;

		pInputData->width = mOutputSize;

		isInitialized = true;

#if _DEBUG
		std::cout << "-----> Success!!" << std::endl;
		printLine();
#endif
	}

	void Affine::setup()
	{
#if _DEBUG
		printLine();
		std::cout << "Affine setup start" << std::endl;
#endif
		assert(isInitialized);

#ifdef GPUA_VAILABLE
		setupParamOnGPU();
#else
		setupParamOnCPU();
#endif // GPUA_VAILABLE

#if _DEBUG
		std::cout << "-----> Success!!" << std::endl;
		printLine();
#endif
	}

	void Affine::setupParamOnCPU()
	{
		pParametersOnCPU.resize(2);

		//Affine/dAffineパラメータ
		//(1)参照
		paramMemory& affineParam = pParametersOnCPU[0];
		paramMemory& affineDParam = pDParametersOnCPU[0];
		//(2)パラメータのサイズを設定
		affineParam.paramNum = mOutputSize * mInputSize;
		affineDParam.paramNum = mOutputSize * mInputSize;
		//(3)パラメータ用の領域確保
		affineParam.paramAddress = new parameterType[affineParam.paramNum];
		affineDParam.paramAddress = new parameterType[affineDParam.paramNum];
		//(4)初期化
		{
			std::random_device seed_gen;
			std::default_random_engine engine(seed_gen());
			std::normal_distribution<> dist(0.0, std::sqrt(2.0 / mInputSize));
			for (u32 idx = 0; idx < affineParam.paramNum; idx++)
			{
				affineParam.paramAddress[idx] = mAffineParamWeight * static_cast<f32>(dist(engine)) / std::sqrt(2.0f / mInputSize);
			}

			for (u32 idx = 0; idx < affineDParam.paramNum; idx++)
			{
				affineDParam.paramAddress[idx] = 0.0f;
			}
		}
		//Biasパラメータ
		//(1)参照
		paramMemory& biasParam = pParametersOnCPU[1];
		paramMemory& biasDParam = pDParametersOnCPU[1];
		//(2)パラメータのサイズを設定
		biasParam.paramNum = mOutputSize;
		biasDParam.paramNum = mOutputSize;
		//(3)パラメータ用の領域確保
		biasParam.paramAddress = new parameterType[biasParam.paramNum];
		biasDParam.paramAddress = new parameterType[biasDParam.paramNum];
		//(4)初期化
		{
			for (u32 idx = 0; idx < biasParam.paramNum; idx++)
			{
				biasParam.paramAddress[idx] = 0.0f;
			}

			for (u32 idx = 0; idx < biasDParam.paramNum; idx++)
			{
				biasDParam.paramAddress[idx] = 0.0f;
			}
		}
	}

	void Affine::terminate()
	{

	}

	void Affine::forward(flowDataType** ppFlowData)
	{
#ifdef GPUA_VAILABLE
		forwardOnGPU(ppFlowData);
#else
		forwardOnCPU(ppFlowDataType);
#endif	

	}

	void Affine::backward(flowDataType**)
	{
#ifdef GPUA_VAILABLE
		backwardOnGPU();
#else
		backwardOnCPU();
#endif	
	}

	void Affine::memcpyHostToDevice()
	{

	}

	void Affine::memcpyDeviceToHost()
	{

	}




	//////////////////////////////////////
	//CPU 関数
	//////////////////////////////////////

	void Affine::forwardOnCPU(flowDataType**)
	{

	}

	void Affine::backwardOnCPU()
	{

	}
} 