#include "../../setting.h"
#include "Adam.h"


namespace miduho::optimizer
{
	void Adam::optimize(layer::BaseLayer& pLayer)
	{
#ifdef GPU_AVAILABLE
		optimizeOnGPU(pLayer);
#else
		optimizeOnCPU(pLayer);
#endif
	}

	void Adam::optimizeOnCPU(layer::BaseLayer& pLayer)
	{

	}
}