#include "Adam.h"

namespace Aoba
{
	namespace optimizer
	{
		void Adam::optimizeOnGPU(std::unique_ptr<layer::BaseLayer>& pLayer)
		{
			std::vector<paramMemory>& param = getLayerParamOnGPU(pLayer);
			std::vector<paramMemory>& dParam = getLayerDParamOnGPU(pLayer);
		}
	}
}