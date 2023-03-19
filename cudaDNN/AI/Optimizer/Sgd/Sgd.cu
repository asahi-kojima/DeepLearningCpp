#include "Sgd.h"

namespace Aoba
{
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
				for (u32 i = 0; i < param.size; i++)
				{
					param.address[i] -= mLearningRate * dParam.address[i];
				}
			}
		}
	}
}