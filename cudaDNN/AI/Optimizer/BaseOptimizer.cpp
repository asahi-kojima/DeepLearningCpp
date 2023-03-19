#include "BaseOptimizer.h"
#include "../Layer/BaseLayer.h"


namespace Aoba
{
	std::vector<paramMemory>& Aoba::optimizer::BaseOptimizer::getLayerParamOnCPU(std::unique_ptr<Aoba::layer::BaseLayer>& pLayer)
	{
		return	pLayer->pParametersOnCPU;
	}
	std::vector<paramMemory>& Aoba::optimizer::BaseOptimizer::getLayerDParamOnCPU(std::unique_ptr<Aoba::layer::BaseLayer>& pLayer)
	{
		return	pLayer->pDParametersOnCPU;
	}

	std::vector<paramMemory>& Aoba::optimizer::BaseOptimizer::getLayerParamOnGPU(std::unique_ptr<Aoba::layer::BaseLayer>& pLayer)
	{
		return	pLayer->pParametersOnGPU;
	}
	std::vector<paramMemory>& Aoba::optimizer::BaseOptimizer::getLayerDParamOnGPU(std::unique_ptr<Aoba::layer::BaseLayer>& pLayer)
	{
		return	pLayer->pDParametersOnGPU;
	}

}