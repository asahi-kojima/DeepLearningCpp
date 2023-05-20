#pragma once
#include "../../../common.h"
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
				std::cout << "Affine Layer" << std::endl;
				std::cout << "	InputSize         = " << mInputSize << std::endl;
				std::cout << "	OutputSize        = " << mOutputSize << std::endl;
				std::cout << "	AffineParamWeight = " << mAffineParamWeight << std::endl;
			}

		private:
			u32 mOutputSize;
			u32 mInputSize;
			u32 mBatchSize;
			f32 mAffineParamWeight = 0.01f;


			u32 mFunc0CallCnt = 0;
			f32 mFunc0AveTime = 0.0f;

			u32 mFunc1CallCnt = 0;
			f32 mFunc1AveTime = 0.0f;

			u32 mWhich = 0;
			bool mNowComparing = true;
			const u32 CaptureTimes = 100;
		};

	}
}