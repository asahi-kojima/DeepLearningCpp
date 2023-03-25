#pragma once


#include "../../../setting.h"
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
			void setupLayerInfo(DataShape*) override;
			

			void memcpyHostToDevice() override;
			void memcpyDeviceToHost() override;

			void initializeOnCPU() override;
			void forwardOnCPU()  override;
			void backwardOnCPU() override;
			void terminateOnCPU() override;

			void initializeOnGPU() override;
			void forwardOnGPU()  override;
			void backwardOnGPU() override;
			void terminateOnGPU() override;


		private:
			u32 mOutputSize;
			u32 mInputSize;
			u32 mBatchSize;

			DataMemory mMask;
		};

	}
}