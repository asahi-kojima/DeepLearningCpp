#include <random>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <thread>
#include <chrono>
#include "Affine.h"
#include "../../../common.h"

namespace Aoba::layer
{


	Affine::Affine(u32 outputSize, f32 weight)
		:mBatchSize(0)
		, mInputSize(0)
		, mOutputSize(outputSize)
		, mOutputShape{ 1,1,outputSize }
		, mAffineParamWeight(weight)
	{
	}

	Affine::Affine(u32 outputC, u32 outputH, u32 outputW, f32 weight)
		:mBatchSize(0)
		, mInputSize(0)
		, mOutputSize(outputC* outputH* outputW)
		, mOutputShape{ outputC, outputH, outputW }
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
#if TIME_DEBUG
		{
			std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif

			for (u32 N = 0; N < mBatchSize; N++)
			{
				for (u32 o = 0; o < mOutputSize; o++)
				{
					f32 result = 0.0f;
					for (u32 i = 0; i < mInputSize; i++)
					{
						result += A(o, i) * I(N, i);
					}
					mForwardResultOnCPU(N, o) = result + b[o];
				}
			}
#if TIME_DEBUG
			f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
			std::string name = "";
			(name += __FUNCTION__) += " : forward";
			timers[name] = elapsedTime;
		}
#endif
	}

	void Affine::backwardOnCPU()
	{
		auto& I = *mInputDataOnCPU;
		auto& dI = *mDInputDataOnCPU;

		auto& A = mParametersPtrOnCPU[0];
		auto& dA = mDParametersPtrOnCPU[0];
		auto& db = mDParametersPtrOnCPU[1];
#if TIME_DEBUG
		{
			std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
			for (u32 o = 0; o < mOutputSize; o++)
			{
				for (u32 i = 0; i < mInputSize; i++)
				{
					f32 result = 0.0f;
					for (u32 N = 0; N < mBatchSize; N++)
					{

						result += I(N, i) * dI(N, o);
					}

					dA(o, i) = result;
				}

				f32 result = 0.0f;
				for (u32 N = 0; N < mBatchSize; N++)
				{
					result += dI(N, o);
				}

				db[o] = result;
			}
#if TIME_DEBUG
			f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
			std::string name = "";
			(name += __FUNCTION__) += " : backward dA db";
			timers[name] = elapsedTime;
		}
#endif

#if TIME_DEBUG
		{
			std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
			for (u32 N = 0; N < mBatchSize; N++)
			{
				for (u32 i = 0; i < mInputSize; i++)
				{
					f32 result = 0.0f;
					for (u32 o = 0; o < mOutputSize; o++)
					{
						result += A(o, i) * dI(N, o);
					}
					mBackwardResultOnCPU(N, i) = result;
				}
			}
#if TIME_DEBUG
			f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
			std::string name = "";
			(name += __FUNCTION__) += " : backward dout";
			timers[name] = elapsedTime;
		}
#endif
	}

	void Affine::terminateOnCPU()
	{
		delete[] mParametersPtrOnCPU[0].address;
		delete[] mParametersPtrOnCPU[1].address;

		delete[] mDParametersPtrOnCPU[0].address;
		delete[] mDParametersPtrOnCPU[1].address;
	}
}