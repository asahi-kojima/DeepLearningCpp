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
			virtual void optimize(std::unique_ptr<layer::BaseLayer>&) = 0;
			virtual void setIsGpuAvailable(bool which) final
			{
				mIsGpuAvailable = which;
			}
			f32 mLearningRate;
		protected:
			bool mIsGpuAvailable = false;
			virtual void optimizeOnCPU(std::unique_ptr<layer::BaseLayer>&) = 0;
			virtual void optimizeOnGPU(std::unique_ptr<layer::BaseLayer>&) = 0;

			std::vector<paramMemory>& getLayerParamOnCPU(std::unique_ptr<layer::BaseLayer>&);
			std::vector<paramMemory>& getLayerDParamOnCPU(std::unique_ptr<layer::BaseLayer>&);

			std::vector<paramMemory>& getLayerParamOnGPU(std::unique_ptr<layer::BaseLayer>&);
			std::vector<paramMemory>& getLayerDParamOnGPU(std::unique_ptr<layer::BaseLayer>&);
		};
	}
}