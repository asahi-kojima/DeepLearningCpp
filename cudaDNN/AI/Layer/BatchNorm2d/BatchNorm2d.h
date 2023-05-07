#pragma once


#include "../../../setting.h"
#include "../BaseLayer.h"


namespace Aoba
{
	namespace layer
	{

		class BatchNorm2d : public BaseLayer
		{
		public:
			BatchNorm2d();
			~BatchNorm2d();

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


		private:
			u32 mBatchSize;
			DataShape mDataShape;

			DataMemory mIntermediateResultOnCPU;
			DataMemory mSigmaOnCPU;
			
			DataMemory mIntermediateResultOnGPU;
			DataMemory mSigmaOnGPU;
		};

	}
}