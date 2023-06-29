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

		private:
			void initializeOnCPU(std::vector<std::unique_ptr<layer::BaseLayer> >&) override {};
			void initializeOnGPU(std::vector<std::unique_ptr<layer::BaseLayer> >&) override {};
			void terminateOnCPU() override {};
			void terminateOnGPU() override {};
			void optimizeOnCPU(std::unique_ptr<layer::BaseLayer>&) override;
			void optimizeOnGPU(std::unique_ptr<layer::BaseLayer>&) override;
		};
	}
}