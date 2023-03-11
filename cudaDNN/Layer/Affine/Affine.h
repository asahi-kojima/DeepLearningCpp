#pragma once
#include "setting.h"
#include "BaseLayer.h"


namespace miduho
{
	namespace layer
	{

		class Affine : public BaseLayer
		{
		public:
			Affine(u32);
			~Affine();
			void initialize(flowDataFormat*) override;
			void setup() override;
			void terminate() override;

			void forward(flowDataType**) override;
			void backward(flowDataType**) override;
			void memcpyHostToDevice() override;
			void memcpyDeviceToHost() override;

		private:
			void forwardOnGPU(flowDataType**)  override;
			void forwardOnCPU(flowDataType**)  override;
			void backwardOnGPU() override;
			void backwardOnCPU() override;

			void setupParamOnCPU();
			void setupParamOnGPU();


		private:
			u32 mOutputSize;
			u32 mInputSize;
			u32 mBatchSize;
			f32 mAffineParamWeight = 0.01f;
		};

	}
}