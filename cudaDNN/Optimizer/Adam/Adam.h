#pragma once

#include "../BaseOptimizer.h"

namespace miduho
{
	namespace optimizer
	{
		class Adam : public BaseOptimizer
		{
		public:
			Adam() = default;
			~Adam() = default;

		private:
			void optimize(layer::BaseLayer&) override;
			void optimizeOnCPU(layer::BaseLayer&) override;
			void optimizeOnGPU(layer::BaseLayer&) override;
		};
	}
}