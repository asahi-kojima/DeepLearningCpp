#pragma once
#include <memory>
#include <vector>
#include "../AIDataStructure.h"
#include "../AIMacro.h"

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

			virtual void initializeOnCPU(std::vector<std::unique_ptr<layer::BaseLayer> >&) = 0;
			virtual void initializeOnGPU(std::vector<std::unique_ptr<layer::BaseLayer> >&) = 0;
			
			virtual void terminateOnCPU() = 0;
			virtual void terminateOnGPU() = 0;

			virtual void optimizeOnCPU(std::unique_ptr<layer::BaseLayer>&) = 0;
			virtual void optimizeOnGPU(std::unique_ptr<layer::BaseLayer>&) = 0;
			virtual void setLearningRate(f32 lr) final {mLearningRate = lr;}

		protected:
			f32 mLearningRate;

			std::vector<DataArray>& getLayerParamOnCPU(std::unique_ptr<layer::BaseLayer>&);
			std::vector<DataArray>& getLayerDParamOnCPU(std::unique_ptr<layer::BaseLayer>&);

			std::vector<DataArray>& getLayerParamOnGPU(std::unique_ptr<layer::BaseLayer>&);
			std::vector<DataArray>& getLayerDParamOnGPU(std::unique_ptr<layer::BaseLayer>&);
		};
	}
}