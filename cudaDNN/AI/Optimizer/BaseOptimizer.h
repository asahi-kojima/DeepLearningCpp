#pragma once
#include <memory>
#include <vector>
#include "../AISetting.h"

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
			BaseOptimizer(f32 learningRate = 0.001f) : mLearningRate(learningRate) {};
			~BaseOptimizer() = default;
			
			virtual void optimizeOnCPU(std::unique_ptr<layer::BaseLayer>&) = 0;
			virtual void optimizeOnGPU(std::unique_ptr<layer::BaseLayer>&) = 0;
			virtual void setLearningRate(f32 lr) final {mLearningRate = lr;}

		protected:
			f32 mLearningRate;

			std::vector<paramMemory>& getLayerParamOnCPU(std::unique_ptr<layer::BaseLayer>&);
			std::vector<paramMemory>& getLayerDParamOnCPU(std::unique_ptr<layer::BaseLayer>&);

			std::vector<paramMemory>& getLayerParamOnGPU(std::unique_ptr<layer::BaseLayer>&);
			std::vector<paramMemory>& getLayerDParamOnGPU(std::unique_ptr<layer::BaseLayer>&);
		};
	}
}