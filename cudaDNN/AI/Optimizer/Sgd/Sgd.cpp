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
	}
}