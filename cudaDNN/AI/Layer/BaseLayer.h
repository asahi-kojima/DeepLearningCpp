#pragma once
#include <vector>
#include <cassert>

#include "../AISetting.h"
#include "../Optimizer/BaseOptimizer.h"
#include <iostream>

namespace Aoba
{
	namespace layer
	{
		class BaseLayer
		{
			friend class Aoba::optimizer::BaseOptimizer;

		public:
			BaseLayer() = default;
			virtual ~BaseLayer() = default;

			virtual void setInputDataOnCPU(DataMemory*& pInputData) final
			{
				mInputDataOnCPU = pInputData;
				pInputData = &mForwardResultOnCPU;
			}
			virtual void setDInputDataOnCPU(DataMemory*& pDInputData) final
			{
				mDInputDataOnCPU = pDInputData;
				pDInputData = &mBackwardResultOnCPU;
			}

			virtual void setInputDataOnGPU(DataMemory*& pInputData) final
			{
				mInputDataOnGPU = pInputData;
				pInputData = &mForwardResultOnGPU;
			}
			virtual void setDInputDataOnGPU(DataMemory*& pDInputData) final
			{
				mDInputDataOnGPU = pDInputData;
				pDInputData = &mBackwardResultOnGPU;
			}

			virtual void printLayerInfo()
			{
				std::cout << "layer information" << std::endl;
			}

			//CPU
			std::vector<paramMemory> pParametersOnCPU;
			std::vector<paramMemory> pDParametersOnCPU;
			DataMemory mForwardResultOnCPU;
			DataMemory mBackwardResultOnCPU;
			DataMemory* mInputDataOnCPU;
			DataMemory* mDInputDataOnCPU;

			virtual void initializeOnCPU(u32 batchSize, DataShape& shape) final
			{
				if (!mIsSetupLayerInfo)
				{
					setupLayerInfo(batchSize, shape);
					mIsSetupLayerInfo = true;
				}
				mallocOnCPU();
			}
			virtual void mallocOnCPU() = 0;

			virtual void forwardOnCPU() = 0;
			virtual void backwardOnCPU() = 0;
			virtual void terminateOnCPU() = 0;


			//GPU
			std::vector<paramMemory> pParametersOnGPU;
			std::vector<paramMemory> pDParametersOnGPU;
			DataMemory mForwardResultOnGPU;
			DataMemory mBackwardResultOnGPU;
			DataMemory* mInputDataOnGPU;
			DataMemory* mDInputDataOnGPU;

			virtual void initializeOnGPU(u32 batchSize, DataShape& shape) final
			{
				if (!mIsSetupLayerInfo)
				{
					setupLayerInfo(batchSize, shape);
					mIsSetupLayerInfo = true;
				}
				mallocOnGPU();
			}
			virtual void mallocOnGPU() = 0;

			virtual void forwardOnGPU() = 0;
			virtual void backwardOnGPU() = 0;
			virtual void terminateOnGPU() = 0;

		private:
			virtual void setupLayerInfo(u32, DataShape&) = 0;
			bool mIsSetupLayerInfo = false;
		};


	}
}