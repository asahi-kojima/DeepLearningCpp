#pragma once


namespace miduho
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
			virtual void optimize(layer::BaseLayer&) = 0;
		protected:
			virtual void optimizeOnCPU(layer::BaseLayer&) = 0;
			virtual void optimizeOnGPU(layer::BaseLayer&) = 0;
			
		};
	}
}