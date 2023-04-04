#pragma once
#include <vector>
#include <cassert>

#include "../AISetting.h"
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


			/// <summary>
			/// 入力・出力サイズ、カーネルサイズなどの決定などを行う。
			/// </summary>
			/// <param name=""></param>
			virtual void setupLayerInfo(DataShape*) = 0;





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


			//CPU
			std::vector<paramMemory> pParametersOnCPU;
			std::vector<paramMemory> pDParametersOnCPU;
			DataMemory mForwardResultOnCPU;
			DataMemory mBackwardResultOnCPU;
			DataMemory* mInputDataOnCPU;
			DataMemory* mDInputDataOnCPU;

			virtual void initializeOnCPU() = 0;
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

			virtual void initializeOnGPU() = 0;
			virtual void forwardOnGPU() = 0;
			virtual void backwardOnGPU() = 0;
			virtual void terminateOnGPU() = 0;
		};


	}
}