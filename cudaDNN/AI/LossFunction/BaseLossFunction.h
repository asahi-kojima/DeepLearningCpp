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

			void setupDataShape(DataShape& dataShape) { mDataShape = dataShape; }



			virtual f32 calcLossAndDInputOnCPU() = 0;
			virtual void initializeOnCPU() = 0;
			void setInputOnCPU(DataMemory* pData, DataMemory* pLabel)
			{
				mForwardResultOnCPU = pData;
				mLabelDataOnCPU = pLabel;
			}
			DataMemory* getDInputDataOnCPU()
			{
				return &mDInputDataOnCPU;
			}



			virtual void initializeOnGPU() = 0;
			virtual f32 calcLossAndDInputOnGPU() = 0;
			void setInputOnGPU(DataMemory* pData, DataMemory* pLabel)
			{
				mForwardResultOnGPU = pData;
				mLabelDataOnGPU = pLabel;
			}

			DataMemory* getDInputDataOnGPU()
			{
				return &mDInputDataOnGPU;
			}

		protected:
			DataShape mDataShape;

			DataMemory mLossTblOnCPU;
			DataMemory mDInputDataOnCPU;
			DataMemory* mForwardResultOnCPU;
			DataMemory* mLabelDataOnCPU;

			DataMemory mLossTblOnGPU;
			DataMemory mDInputDataOnGPU;
			DataMemory* mForwardResultOnGPU;
			DataMemory* mLabelDataOnGPU;
		};
	}
}