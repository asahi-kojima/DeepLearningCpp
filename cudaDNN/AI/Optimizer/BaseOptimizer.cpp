#include "BaseOptimizer.h"
#include "../Layer/BaseLayer.h"


namespace Aoba
{
	std::vector<DataArray>& Aoba::optimizer::BaseOptimizer::getLayerParamOnCPU(std::unique_ptr<Aoba::layer::BaseLayer>& pLayer)
	{
		return	pLayer->mParametersPtrOnCPU;
	}
	std::vector<DataArray>& Aoba::optimizer::BaseOptimizer::getLayerDParamOnCPU(std::unique_ptr<Aoba::layer::BaseLayer>& pLayer)
	{
		return	pLayer->mDParametersPtrOnCPU;
	}

	std::vector<DataArray>& Aoba::optimizer::BaseOptimizer::getLayerParamOnGPU(std::unique_ptr<Aoba::layer::BaseLayer>& pLayer)
	{
		return	pLayer->mParametersPtrOnGPU;
	}
	std::vector<DataArray>& Aoba::optimizer::BaseOptimizer::getLayerDParamOnGPU(std::unique_ptr<Aoba::layer::BaseLayer>& pLayer)
	{
		return	pLayer->mDParametersPtrOnGPU;
	}

}