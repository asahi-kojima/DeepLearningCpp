#include <cuda_runtime.h>

#include "Sgd.h"




namespace Aoba
{
	namespace
	{
		__global__ void OptimizeOnGPU(f32* param, f32* dParam, f32 learningRate, u32 size)
		{
			u32 id = blockIdx.x * blockDim.x + threadIdx.x;
			if (id >= size)
			{
				return;
			}
			param[id] -= dParam[id] * learningRate;
		}
	}
	namespace optimizer
	{
		void Sgd::optimizeOnGPU(std::unique_ptr<layer::BaseLayer>& pLayer)
		{
			std::vector<DataArray>& params = getLayerParamOnGPU(pLayer);
			std::vector<DataArray>& dParams = getLayerDParamOnGPU(pLayer);

			for (u32 idx = 0; idx < params.size(); idx++)
			{
				DataArray& param = params[idx];
				DataArray& dParam = dParams[idx];

				dim3 block(32);
				dim3 grid((param.size + block.x - 1) / block.x);

				OptimizeOnGPU << <grid, block >> > (param.address, dParam.address, mLearningRate, param.size);
#if _DEBUG
				CHECK(cudaDeviceSynchronize());
#endif
			}
		}
	}
}