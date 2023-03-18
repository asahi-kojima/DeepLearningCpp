#include "../../setting.h"
#include "Adam.h"
#include "../../Layer/Layer.h"

namespace Aoba::optimizer
{
	void Adam::optimize(std::unique_ptr<layer::BaseLayer>& pLayer)
	{
#ifdef GPU_AVAILABLE
		optimizeOnGPU(pLayer);
#else
		optimizeOnCPU(pLayer);
#endif
	}

	void Adam::optimizeOnCPU(std::unique_ptr<layer::BaseLayer>& pLayer)
	{
		//auto x = pLayer->pParametersOnCPU;
		//std::vector<layer::BaseLayer::paramMemory>& layer = fff(pLayer);
	}
}