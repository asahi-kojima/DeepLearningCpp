#include <random>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

#include "Affine.h"
#include "../../../commonCPU.h"

namespace Aoba::layer
{


	Affine::Affine(u32 outputSize, f32 weight)
		:mBatchSize(0)
		,mInputSize(0)
		,mOutputSize(outputSize)
		,mAffineParamWeight(weight)
	{
	}

	Affine::~Affine()
	{
	}

	void Affine::setupLayerInfo(u32 batchSize, DataShape& shape)
	{
		mBatchSize = batchSize;
		mInputSize = shape.width;

		shape.width = mOutputSize;
	}





	//////////////////////////////////////
	//CPU 関数
	//////////////////////////////////////
	void Affine::mallocOnCPU()
	{
		pParametersOnCPU.resize(2);
		pDParametersOnCPU.resize(2);

		//Affine/dAffineパラメータ
		//(1)参照
		paramMemory& affineParam = pParametersOnCPU[0];
		paramMemory& affineDParam = pDParametersOnCPU[0];
		//(2)パラメータのサイズを設定
		affineParam.size = mOutputSize * mInputSize;
		affineDParam.size = mOutputSize * mInputSize;
		//(3)パラメータ用の領域確保
		affineParam.address = new f32[affineParam.size];
		affineDParam.address = new f32[affineDParam.size];
		//(4)初期化
		{
			std::random_device seed_gen;
			std::default_random_engine engine(seed_gen());
			std::normal_distribution<> dist(0.0, std::sqrt(2.0 / mInputSize));
			for (u32 idx = 0; idx < affineParam.size; idx++)
			{
				affineParam.address[idx] = mAffineParamWeight * static_cast<f32>(dist(engine)) / std::sqrt(2.0f / mInputSize);
			}

			for (u32 idx = 0; idx < affineDParam.size; idx++)
			{
				affineDParam.address[idx] = 0.0f;
			}
		}
		//Biasパラメータ
		//(1)参照
		paramMemory& biasParam = pParametersOnCPU[1];
		paramMemory& biasDParam = pDParametersOnCPU[1];
		//(2)パラメータのサイズを設定
		biasParam.size = mOutputSize;
		biasDParam.size = mOutputSize;
		//(3)パラメータ用の領域確保
		biasParam.address = new f32[biasParam.size];
		biasDParam.address = new f32[biasDParam.size];
		//(4)初期化
		{
			for (u32 idx = 0; idx < biasParam.size; idx++)
			{
				biasParam.address[idx] = 0.0f;
			}

			for (u32 idx = 0; idx < biasDParam.size; idx++)
			{
				biasDParam.address[idx] = 0.0f;
			}
		}

		mForwardResultOnCPU.size = mBatchSize * mOutputSize;
		mBackwardResultOnCPU.size = mBatchSize * mInputSize;

		mForwardResultOnCPU.address = new f32[mForwardResultOnCPU.size];
		mBackwardResultOnCPU.address = new f32[mBackwardResultOnCPU.size];
	}

	void Affine::forwardOnCPU()
	{
		for (u32 N = 0; N < mBatchSize; N++)
		{
			for (u32 o = 0; o < mOutputSize; o++)
			{
				u32 index = N * mOutputSize + o;
				f32 result = 0.0f;
				for (u32 i = 0; i < mInputSize; i++)
				{
#if _DEBUG
					assert(pParametersOnCPU[0].size > o * mInputSize + i);
					assert(mInputDataOnCPU->size > N * mInputSize + i);
#endif
					result += pParametersOnCPU[0].address[o * mInputSize + i] * mInputDataOnCPU->address[N * mInputSize + i];
				}
#if _DEBUG
				assert(mForwardResultOnCPU.size > index);
				assert(pParametersOnCPU[1].size > o);
#endif
				mForwardResultOnCPU.address[index] = result + pParametersOnCPU[1].address[o];
			}
		}
	}

	void Affine::backwardOnCPU()
	{
		for (u32 o = 0; o < mOutputSize; o++)
		{
			for (u32 i = 0; i < mInputSize; i++)
			{
				f32 result = 0.0f;
				for (u32 N = 0; N < mBatchSize; N++)
				{
#if _DEBUG
					assert(mInputDataOnCPU->size > N * mInputSize + i);
					assert(mDInputDataOnCPU->size > N * mOutputSize + o);
#endif
					result += mInputDataOnCPU->address[N * mInputSize + i] * mDInputDataOnCPU->address[N * mOutputSize + o];
				}
#if _DEBUG
				assert(pDParametersOnCPU[0].size > o * mInputSize + i);
#endif
				pDParametersOnCPU[0].address[o * mInputSize + i] = result;
			}

			f32 result = 0.0f;
			for (u32 N = 0; N < mBatchSize; N++)
			{
#if _DEBUG
				assert(mDInputDataOnCPU->size > N * mOutputSize + o);
#endif
				result += mDInputDataOnCPU->address[N * mOutputSize + o];
			}
#if _DEBUG
			assert(pDParametersOnCPU[1].size > o);
#endif
			pDParametersOnCPU[1].address[o] = result;
		}

		for (u32 N = 0; N < mBatchSize; N++)
		{
			for (u32 i = 0; i < mInputSize; i++)
			{
				f32 result = 0.0f;
				for (u32 o = 0; o < mOutputSize; o++)
				{
#if _DEBUG
					assert(pParametersOnCPU[0].size > o * mInputSize + i);
					assert(mDInputDataOnCPU->size > N * mOutputSize + o);
#endif
					result += pParametersOnCPU[0].address[o * mInputSize + i] * mDInputDataOnCPU->address[N * mOutputSize + o];
				}
#if _DEBUG
				assert(mBackwardResultOnCPU.size > N * mInputSize + i);
#endif
				mBackwardResultOnCPU.address[N * mInputSize + i] = result;
			}
		}
	}

	void Affine::terminateOnCPU()
	{
		delete[] pParametersOnCPU[0].address;
		delete[] pParametersOnCPU[1].address;

		delete[] pDParametersOnCPU[0].address;
		delete[] pDParametersOnCPU[1].address;
	}
} 

#if _DEBUG
#endif