#pragma once

#include "../BaseOptimizer.h"

namespace Aoba
{
	namespace optimizer
	{
		class Sgd : public BaseOptimizer
		{
		public:
			Sgd(f32 learningRate = 0.001f) : BaseOptimizer(learningRate) {}
			~Sgd() = default;

			void optimize(std::unique_ptr<layer::BaseLayer>&) override;

		private:
			void optimizeOnCPU(std::unique_ptr<layer::BaseLayer>&) override;
			void optimizeOnGPU(std::unique_ptr<layer::BaseLayer>&) override;
		};
	}
}