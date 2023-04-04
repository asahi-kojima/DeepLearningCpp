#pragma once

#include "../BaseOptimizer.h"

namespace Aoba
{
	namespace optimizer
	{
		class Adam : public BaseOptimizer
		{
		public:
			Adam() = default;
			~Adam() = default;

		
		private:
			void optimizeOnCPU(std::unique_ptr<layer::BaseLayer>&) override;
			void optimizeOnGPU(std::unique_ptr<layer::BaseLayer>&) override;
		};
	}
}