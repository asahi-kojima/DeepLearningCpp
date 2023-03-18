#pragma once
#include "../../../setting.h"
#include "../BaseLayer.h"


namespace Aoba
{
	namespace layer
	{

		class Affine : public BaseLayer
		{
		public:
			Affine(u32);
			~Affine();
			
		private:
			void setupLayerInfo(DataShape*) override;

			void initialize() override;
			void forward() override;
			void backward() override;
			void terminate() override;

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
			f32 mAffineParamWeight = 0.01f;
		};

	}
}