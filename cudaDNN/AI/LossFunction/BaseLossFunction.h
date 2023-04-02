#pragma once

#include "../AISetting.h"
#include "../Layer/BaseLayer.h"
namespace Aoba
{
	namespace lossFunction
	{
		class BaseLossFunction
		{
		public:
			BaseLossFunction() = default;
			~BaseLossFunction() = default;

			void setupDataShape(layer::BaseLayer::DataShape& dataShape) { mDataShape = dataShape; }
			void initialize()
			{
				if (mIsGpuAvailable)
				{
					initializeOnGPU();
#if _DEBUG
					initializeOnCPU();
#endif
				}
				else
				{
					initializeOnCPU();
				}
			}

			f32 calcLossAndDInput()
			{
				if (mIsGpuAvailable)
				{
#if _DEBUG
					calcLossAndDInputOnCPU();
#endif
					return calcLossAndDInputOnGPU();
				}
				else
				{
					return calcLossAndDInputOnCPU();
				}
			}


			void setInput(DataMemory* pData, DataMemory* pLabel)
			{
				mForwardResult = pData;
				mLabelData = pLabel;
			}


			void setInputForGpuDebug(DataMemory* pData, DataMemory* pLabel)
			{
				mForwardResultForGpuDebug = pData;
				mLabelDataForGpuDebug = pLabel;
			}


			DataMemory mDInputData;
#if _DEBUG
			DataMemory mDInputDataForGpuDebug;
#endif

			virtual void setIsGpuAvailable(bool which) final
			{
				mIsGpuAvailable = which;
			}


		protected:
			bool mIsGpuAvailable = false;

			layer::BaseLayer::DataShape mDataShape;
			virtual void initializeOnGPU()=0;
			virtual void initializeOnCPU()=0;

			virtual f32 calcLossAndDInputOnGPU() = 0;
			virtual f32 calcLossAndDInputOnCPU() = 0;


			DataMemory* mForwardResult;
			DataMemory* mLabelData;
#if _DEBUG
			DataMemory* mForwardResultForGpuDebug;
			DataMemory* mLabelDataForGpuDebug;
#endif
		};
	}
}