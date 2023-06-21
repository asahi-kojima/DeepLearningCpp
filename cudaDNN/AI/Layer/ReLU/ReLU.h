#pragma once
#include "../BaseLayer.h"


namespace Aoba
{
	namespace layer
	{

		class ReLU : public BaseLayer
		{
		public:
			ReLU();
			~ReLU();

		private:
			void setupLayerInfo(u32, DataShape&) override;
			

			void mallocOnCPU() override;
			void forwardOnCPU()  override;
			void backwardOnCPU() override;
			void terminateOnCPU() override;

			void mallocOnGPU() override;
			void forwardOnGPU()  override;
			void backwardOnGPU() override;
			void terminateOnGPU() override;

			void printLayerInfo() override
			{
				printDoubleLine();
				printLayerName("ReLU Layer");
				print3dProperty("InputSize", mDataShape);
				print3dProperty("OutputSize", mDataShape);
			}

		private:
			static u32 InstanceCounter;
			u32 mInstanceID;

			u32 mBatchSize;
			DataShape mDataShape;
			u32 mDataSize;

			DataArray mMaskOnCPU;
			DataArray mMaskOnGPU;
		};

	}
}