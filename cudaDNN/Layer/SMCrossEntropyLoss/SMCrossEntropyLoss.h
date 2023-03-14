#pragma once
#include "../../setting.h"
#include "../BaseLayer.h"


namespace miduho
{
	namespace layer
	{

		class SMCrossEntropyLoss : public BaseLayer
		{
		public:
			SMCrossEntropyLoss(u32);
			~SMCrossEntropyLoss();

			void setupLayerInfo(FlowDataFormat*) override;

			void initialize() override;
			void forward() override;
			void backward() override;
			void terminate() override;

			void memcpyHostToDevice() override;
			void memcpyDeviceToHost() override;

		private:
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
		};

	}
}