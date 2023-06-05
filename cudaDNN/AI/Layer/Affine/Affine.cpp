#include <random>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

#include "Affine.h"
#include "../../../common.h"

namespace Aoba::layer
{


	Affine::Affine(u32 outputSize, f32 weight)
		:mBatchSize(0)
		,mInputSize(0)
		,mOutputSize(outputSize)
		,mOutputShape{1,1,outputSize}
		,mAffineParamWeight(weight)
	{
	}

	Affine::Affine(u32 outputC, u32 outputH, u32 outputW, f32 weight)
		:mBatchSize(0)
		, mInputSize(0)
		, mOutputSize(outputC * outputH * outputW)
		, mOutputShape{outputC, outputH, outputW}
		, mAffineParamWeight(weight)
	{
	}

	Affine::~Affine()
	{
	}

	void Affine::setupLayerInfo(u32 batchSize, DataShape& shape)
	{
		mBatchSize = batchSize;

		mInputShape = shape;
		mInputSize = shape.getDataSize();

		shape = mOutputShape;
	}





	//////////////////////////////////////
	//CPU 関数
	//////////////////////////////////////
	void Affine::mallocOnCPU()
	{
		mParametersPtrOnCPU.resize(2);
		mDParametersPtrOnCPU.resize(2);

		//Affine/dAffineパラメータ
		//(1)参照
		DataArray& affineParam = mParametersPtrOnCPU[0];
		DataArray& affineDParam = mDParametersPtrOnCPU[0];
		//(2)パラメータのサイズを設定
		affineParam.setSizeAs2D(mOutputSize, mInputSize);
		affineDParam.setSizeAs2D(mOutputSize, mInputSize);
		//(3)パラメータ用の領域確保と初期化
		MALLOC_AND_INITIALIZE_NORMAL_ON_CPU(affineParam, mInputSize, mAffineParamWeight);
		MALLOC_AND_INITIALIZE_0_ON_CPU(affineDParam);

		//Biasパラメータ
		//(1)参照
		DataArray& biasParam = mParametersPtrOnCPU[1];
		DataArray& biasDParam = mDParametersPtrOnCPU[1];
		//(2)パラメータのサイズを設定
		biasParam.size = biasDParam.size = mOutputSize;
		//(3)パラメータ用の領域確保と初期化
		MALLOC_AND_INITIALIZE_0_ON_CPU(biasParam);
		MALLOC_AND_INITIALIZE_0_ON_CPU(biasDParam);


		//伝搬用変数
		mForwardResultOnCPU.setSizeAs2D(mBatchSize, mOutputSize);
		mBackwardResultOnCPU.setSizeAs4D(mBatchSize, mInputShape);

		MALLOC_AND_INITIALIZE_0_ON_CPU(mForwardResultOnCPU);
		MALLOC_AND_INITIALIZE_0_ON_CPU(mBackwardResultOnCPU);
	}

	void Affine::forwardOnCPU()
	{
		auto& I = *mInputDataOnCPU;
		auto& A = mParametersPtrOnCPU[0];
		auto& b = mParametersPtrOnCPU[1];
		for (u32 N = 0; N < mBatchSize; N++)
		{
			for (u32 o = 0; o < mOutputSize; o++)
			{
				f32 result = 0.0f;
				for (u32 i = 0; i < mInputSize; i++)
				{
#if _DEBUG
					assert(A.size > o * mInputSize + i);
					assert(I.size > N * mInputSize + i);
#endif
					result += A(o, i) * I(N, i);
				}
#if _DEBUG
				assert(mForwardResultOnCPU.size > N * mOutputSize + o);
				assert(b.size > o);
#endif
				mForwardResultOnCPU(N, o) = result + b[o];
			}
		}
	}

	void Affine::backwardOnCPU()
	{
		auto& I = *mInputDataOnCPU;
		auto& dI = *mDInputDataOnCPU;

		auto& A = mParametersPtrOnCPU[0];
		auto& dA = mDParametersPtrOnCPU[0];
		auto& db = mDParametersPtrOnCPU[1];

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
					result += I(N, i) * dI(N, o);
				}
#if _DEBUG
				assert(dA.size > o * mInputSize + i);
#endif
				dA(o, i) = result;
			}

			f32 result = 0.0f;
			for (u32 N = 0; N < mBatchSize; N++)
			{
#if _DEBUG
				assert(dI.size > N * mOutputSize + o);
#endif
				result += dI(N, o);
			}
#if _DEBUG
			assert(db.size > o);
#endif
			db[o] = result;
		}

		for (u32 N = 0; N < mBatchSize; N++)
		{
			for (u32 i = 0; i < mInputSize; i++)
			{
				f32 result = 0.0f;
				for (u32 o = 0; o < mOutputSize; o++)
				{
#if _DEBUG
					assert(mParametersPtrOnCPU[0].size > o * mInputSize + i);
					assert(mDInputDataOnCPU->size > N * mOutputSize + o);
#endif
					result += A(o, i) * dI(N, o);
				}
#if _DEBUG
				assert(mBackwardResultOnCPU.size > N * mInputSize + i);
#endif
				mBackwardResultOnCPU(N, i) = result;
			}
		}
	}

	void Affine::terminateOnCPU()
	{
		delete[] mParametersPtrOnCPU[0].address;
		delete[] mParametersPtrOnCPU[1].address;

		delete[] mDParametersPtrOnCPU[0].address;
		delete[] mDParametersPtrOnCPU[1].address;
	}
}