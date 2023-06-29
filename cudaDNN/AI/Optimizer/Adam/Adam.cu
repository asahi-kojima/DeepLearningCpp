#include <cuda_runtime.h>
#include "Adam.h"
#include "../../AIHelperFunction.h"

namespace Aoba
{
	namespace
	{
		__global__ void OptimizeOnGPU(f32* param, f32* dParam, f32* m, f32* v, f32 beta0, f32 beta1, f32 learningRate, u32 size)
		{
			u32 id = blockIdx.x * blockDim.x + threadIdx.x;
			if (id >= size)
			{
				return;
			}

			auto dP = dParam[id];
			f32 tmpM = m[id] += (1 - beta0) * (dP - m[id]);
			f32 tmpV = v[id] += (1 - beta1) * (dP * dP - v[id]);

			param[id] -= learningRate * tmpM / (sqrtf(tmpV) + 1e-7);
		}
	}


	namespace optimizer
	{
		void Adam::initializeOnGPU(std::vector<std::unique_ptr<layer::BaseLayer> >& layerList)
		{
			const u32 layerSize = layerList.size();

			mMomentumOnGPU.resize(layerSize);
			mVelocityOnGPU.resize(layerSize);

			for (u32 order = 0; order < layerSize; order++)
			{
				auto pLayer = layerList[order].get();
				mLayerOrderMapOnGPU[pLayer] = order;

				std::vector<DataArray>& parameters = getLayerParamOnGPU(layerList[order]);
				mMomentumOnGPU[order].resize(parameters.size());
				mVelocityOnGPU[order].resize(parameters.size());

				for (u32 id = 0; id < mMomentumOnGPU[order].size(); id++)
				{
					mMomentumOnGPU[order][id].size = parameters[id].size;
					mVelocityOnGPU[order][id].size = parameters[id].size;
					MALLOC_AND_INITIALIZE_0_ON_GPU(mMomentumOnGPU[order][id]);
					MALLOC_AND_INITIALIZE_0_ON_GPU(mVelocityOnGPU[order][id]);
				}
			}
		}

		void Adam::optimizeOnGPU(std::unique_ptr<layer::BaseLayer>& refLayer)
		{
			auto pLayer = refLayer.get();
			const u32 order = mLayerOrderMapOnGPU[pLayer];


			std::vector<DataArray>& momentumLst = mMomentumOnGPU[order];
			std::vector<DataArray>& velocityLst = mVelocityOnGPU[order];

			mIteration++;
			f32 effectiveLr = mLearningRate * std::sqrtf(1.0f - std::powf(mBeta1, mIteration)) / (1.0f - std::powf(mBeta0, mIteration));
			for (u32 id = 0; id < momentumLst.size(); id++)
			{
				auto& m = momentumLst[id];
				auto& v = velocityLst[id];

				auto& param = getLayerParamOnGPU(refLayer)[id];
				auto& dParam = getLayerDParamOnGPU(refLayer)[id];

				dim3 block(32);
				dim3 grid((param.size + block.x - 1) / block.x);
#if TIME_DEBUG
				{
					std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
					OptimizeOnGPU << <grid, block >> > (param.address, dParam.address, m.address, v.address, mBeta0, mBeta1, effectiveLr, param.size);
#if GPU_SYNC_DEBUG
					CHECK(cudaDeviceSynchronize());
#endif
#if TIME_DEBUG
					f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
					std::string name = "";
					(((name += __FUNCTION__) += " : ") ) += " : OptimizeOnGPU";
					timers[name] = elapsedTime;
				}
#endif
			}
		}

		void Adam::terminateOnGPU()
		{
			for (u32 order = 0; order < mMomentumOnGPU.size(); order++)
			{
				for (u32 id = 0; id < mMomentumOnGPU[order].size(); id++)
				{
					CUDA_FREE(mMomentumOnGPU[order][id]);
					CUDA_FREE(mVelocityOnGPU[order][id]);
				}
			}
		}
	}
}