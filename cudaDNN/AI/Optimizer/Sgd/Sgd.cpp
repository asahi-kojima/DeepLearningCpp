#include "Sgd.h"
#include "../../Layer/Layer.h"
#include "../../AISetting.h"

namespace Aoba::optimizer
{
	void Sgd::optimize(std::unique_ptr<layer::BaseLayer>& pLayer)
	{
#ifdef GPU_AVAILABLE
		optimizeOnGPU(pLayer);
#else
		optimizeOnCPU(pLayer);
#endif
	}

	void Sgd::optimizeOnCPU(std::unique_ptr<layer::BaseLayer>& pLayer)
	{
		std::vector<paramMemory>& params = getLayerParamOnCPU(pLayer);
		std::vector<paramMemory>& dParams = getLayerDParamOnCPU(pLayer);

		for (u32 idx = 0; idx < params.size(); idx++)
		{
			paramMemory& param = params[idx];
			paramMemory& dParam = dParams[idx];
			for (u32 i = 0; i < param.size; i++)
			{
				param.address[i] -= mLearningRate * dParam.address[i];
			}
		}
	}
}