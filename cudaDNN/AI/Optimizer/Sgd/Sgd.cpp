#include "Sgd.h"
#include "../../Layer/Layer.h"
#include "../../AIDataStructure.h"

namespace Aoba::optimizer
{


	void Sgd::optimizeOnCPU(std::unique_ptr<layer::BaseLayer>& pLayer)
	{
		std::vector<DataArray>& params = getLayerParamOnCPU(pLayer);
		std::vector<DataArray>& dParams = getLayerDParamOnCPU(pLayer);

		for (u32 idx = 0; idx < params.size(); idx++)
		{
			DataArray& param = params[idx];
			DataArray& dParam = dParams[idx];
			for (u32 i = 0; i < param.size; i++)
			{
				auto mp = mLearningRate * dParam.address[i];
				param.address[i] -= mLearningRate * dParam.address[i];
			}
		}
	}
}