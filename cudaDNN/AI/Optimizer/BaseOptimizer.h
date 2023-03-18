#pragma once
#include <memory>
//#include "../Layer/BaseLayer.h"


namespace Aoba
{
	namespace layer
	{
		class BaseLayer;
	}
	namespace optimizer
	{
		class BaseOptimizer
		{
		public:
			BaseOptimizer() = default;
			~BaseOptimizer() = default;
			virtual void optimize(std::unique_ptr<layer::BaseLayer>&) = 0;
		protected:
			virtual void optimizeOnCPU(std::unique_ptr<layer::BaseLayer>&) = 0;
			virtual void optimizeOnGPU(std::unique_ptr<layer::BaseLayer>&) = 0;
		/*	std::vector<layer::BaseLayer::paramMemory> fff(std::unique_ptr<layer::BaseLayer>& pLayer);*/
		};
	}
}