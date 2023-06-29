#pragma once
#include <vector>
#include <cassert>
#include <iostream>
#include  "../AIHelperFunction.h"
#include "../AIDataStructure.h"
#include "../AIMacro.h"
#include "../Optimizer/BaseOptimizer.h"

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

			virtual void setInputDataOnCPU(DataArray*& pInputData) final
			{
				mInputDataOnCPU = pInputData;
				pInputData = &mForwardResultOnCPU;
			}
			virtual void setDInputDataOnCPU(DataArray*& pDInputData) final
			{
				mDInputDataOnCPU = pDInputData;
				pDInputData = &mBackwardResultOnCPU;
			}

			virtual void setInputDataOnGPU(DataArray*& pInputData) final
			{
				mInputDataOnGPU = pInputData;
				pInputData = &mForwardResultOnGPU;
			}
			virtual void setDInputDataOnGPU(DataArray*& pDInputData) final
			{
				mDInputDataOnGPU = pDInputData;
				pDInputData = &mBackwardResultOnGPU;
			}

			virtual void printLayerInfo()
			{
				printDoubleLine();
				std::cout << "No Infomation" << std::endl;
			}

			//CPU
			std::vector<DataArray> mParametersPtrOnCPU;
			std::vector<DataArray> mDParametersPtrOnCPU;
			DataArray mForwardResultOnCPU;
			DataArray mBackwardResultOnCPU;
			DataArray* mInputDataOnCPU;
			DataArray* mDInputDataOnCPU;

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
			std::vector<DataArray> mParametersPtrOnGPU;
			std::vector<DataArray> mDParametersPtrOnGPU;
			DataArray mForwardResultOnGPU;
			DataArray mBackwardResultOnGPU;
			DataArray* mInputDataOnGPU;
			DataArray* mDInputDataOnGPU;

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