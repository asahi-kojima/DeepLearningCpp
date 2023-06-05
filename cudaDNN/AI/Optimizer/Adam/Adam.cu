#include "Adam.h"

namespace Aoba
{
	namespace optimizer
	{
		void Adam::initializeOnGPU(std::vector<std::unique_ptr<layer::BaseLayer> >&)
		{

		}

		void Adam::optimizeOnGPU(std::unique_ptr<layer::BaseLayer>& pLayer)
		{
			std::vector<DataArray>& param = getLayerParamOnGPU(pLayer);
			std::vector<DataArray>& dParam = getLayerDParamOnGPU(pLayer);
		}
	}
}