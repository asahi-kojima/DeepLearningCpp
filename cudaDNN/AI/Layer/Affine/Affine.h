#pragma once
#include "../../AISetting.h"
#include "../BaseLayer.h"


namespace Aoba
{
	namespace layer
	{
		class Affine : public BaseLayer
		{
		public:
			Affine(u32, f32 weight = 0.01f);
			~Affine();
			
		private:
			void setupLayerInfo(DataShape*) override;


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