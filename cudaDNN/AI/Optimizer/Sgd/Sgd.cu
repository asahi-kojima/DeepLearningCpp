#include <cuda_runtime.h>

#include "../../../commonGPU.cuh"
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
			f32 tmp = param[id];
			param[id] -= dParam[id] * learningRate;
			//printf("param[%d] = %lf | dParam[%d] = %lf | pre = %lf\n", id, param[id], id, dParam[id], tmp);
		}
	}
	namespace optimizer
	{
		void Sgd::optimizeOnGPU(std::unique_ptr<layer::BaseLayer>& pLayer)
		{
			std::vector<paramMemory>& params = getLayerParamOnGPU(pLayer);
			std::vector<paramMemory>& dParams = getLayerDParamOnGPU(pLayer);

			for (u32 idx = 0; idx < params.size(); idx++)
			{
				paramMemory& param = params[idx];
				paramMemory& dParam = dParams[idx];

				dim3 block(32);
				dim3 grid((param.size + block.x - 1) / block.x);
#if _DEBUG
				{
					std::vector<f32> tester0(param.size);
					std::vector<f32> tester1(dParam.size);
					CHECK(cudaMemcpy(tester0.data(), param.address, tester0.size() * sizeof(f32), cudaMemcpyDeviceToHost));
					CHECK(cudaMemcpy(tester1.data(), dParam.address, tester1.size() * sizeof(f32), cudaMemcpyDeviceToHost));
					CHECK(cudaDeviceSynchronize());
				}
#endif
				OptimizeOnGPU << <grid, block >> > (param.address, dParam.address, mLearningRate, param.size);
#if _DEBUG
				CHECK(cudaDeviceSynchronize());
				{
					std::vector<f32> tester0(param.size);
					std::vector<f32> tester1(dParam.size);
					CHECK(cudaMemcpy(tester0.data(), param.address, tester0.size() * sizeof(f32), cudaMemcpyDeviceToHost));
					CHECK(cudaMemcpy(tester1.data(), dParam.address, tester1.size() * sizeof(f32), cudaMemcpyDeviceToHost));
					CHECK(cudaDeviceSynchronize());
				}
#endif
			}
		}
	}
}