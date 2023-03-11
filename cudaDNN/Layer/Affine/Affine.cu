#include <random>
#include <cuda_runtime.h>

#include "Affine.h"
#include "../../commonGPU.cuh"

namespace miduho {
	namespace layer
	{
		void Affine::forwardOnGPU(flowDataType** ppFlowData)
		{

		}

		void Affine::backwardOnGPU()
		{

		}

		void Affine::setupParamOnGPU()
		{
			pParametersOnGPU.resize(2);
			pDParametersOnGPU.resize(2);

			//Affineパラメータ
			paramMemory& affineParam = pParametersOnGPU[0];
			paramMemory& affineDParam = pDParametersOnGPU[0];

			affineParam.paramNum = affineDParam.paramNum = mOutputSize * mInputSize;

			CHECK(cudaMalloc((void**)(&(affineParam.paramAddress)), affineParam.paramNum * sizeof(parameterType))   );
			CHECK(cudaMalloc((void**)(&(affineDParam.paramAddress)), affineDParam.paramNum * sizeof(parameterType)) );

			parameterType* tmpAffineParam = new parameterType[affineParam.paramNum];
			{
				std::random_device seed_gen;
				std::default_random_engine engine(seed_gen());
				std::normal_distribution<> dist(0.0, std::sqrt(2.0 / mInputSize));

				parameterType* tmp = new parameterType[affineParam.paramNum];
				for (u32 idx = 0; idx < affineParam.paramNum; idx++)
				{
					tmp[idx] = mAffineParamWeight * static_cast<f32>(dist(engine)) / std::sqrt(2.0f / mInputSize);
				}
				CHECK(cudaMemcpy(affineParam.paramAddress, tmp, affineParam.paramNum * sizeof(parameterType), cudaMemcpyHostToDevice));

				for (u32 idx = 0; idx < affineDParam.paramNum; idx++)
				{
					tmp[idx] = 0.0f;
				}
				CHECK(cudaMemcpy(affineDParam.paramAddress, tmp, affineDParam.paramNum * sizeof(parameterType), cudaMemcpyHostToDevice));
				delete[] tmp;
			}


			//Biasパラメータ
			paramMemory& biasParam = pParametersOnGPU[1];
			paramMemory& biasDParam = pDParametersOnGPU[1];

			biasParam.paramNum = biasDParam.paramNum = mOutputSize;

			cudaMalloc((void**)(&(biasParam.paramAddress)), biasParam.paramNum * sizeof(parameterType));
			cudaMalloc((void**)(&(biasDParam.paramAddress)), biasDParam.paramNum * sizeof(parameterType));
			{
				parameterType* tmp = new parameterType[biasParam.paramNum];
				for (u32 idx = 0; idx < biasParam.paramNum; idx++)
				{
					tmp[idx] = 0.0f;
				}
				CHECK(cudaMemcpy(biasParam.paramAddress, tmp, biasParam.paramNum * sizeof(parameterType), cudaMemcpyHostToDevice));
				CHECK(cudaMemcpy(biasDParam.paramAddress, tmp, biasDParam.paramNum * sizeof(parameterType), cudaMemcpyHostToDevice));
				delete[] tmp;
			}

			//計算結果を格納するためのメモリ確保
			mForwardResultOnGPU.dataNum = mBatchSize * mOutputSize;
			mBackwardResultOnGPU.dataNum = mBatchSize * mInputSize;
			cudaMalloc((void**)(&(mForwardResultOnGPU.dataAddress)), 
				mForwardResultOnGPU.dataNum * sizeof(flowDataType));
			cudaMalloc((void**)(&(mBackwardResultOnGPU.dataAddress)), 
				mBackwardResultOnGPU.dataNum * sizeof(flowDataType));
			{
				flowDataType* tmp = new flowDataType[mForwardResultOnGPU.dataNum];
				for (u32 idx = 0; idx < mForwardResultOnGPU.dataNum; idx++)
				{
					tmp[idx] = 0.0f;
				}
				CHECK(cudaMemcpy(mForwardResultOnGPU.dataAddress, tmp, 
					mForwardResultOnGPU.dataNum * sizeof(flowDataType), cudaMemcpyHostToDevice));
				delete[] tmp;


				tmp = new flowDataType[mBackwardResultOnGPU.dataNum];
				for (u32 idx = 0; idx < mBackwardResultOnGPU.dataNum; idx++)
				{
					tmp[idx] = 0.0f;
				}
				CHECK(cudaMemcpy(mBackwardResultOnGPU.dataAddress, tmp, 
					mBackwardResultOnGPU.dataNum * sizeof(flowDataType), cudaMemcpyHostToDevice));
				delete[] tmp;
			}
		}
	}
}