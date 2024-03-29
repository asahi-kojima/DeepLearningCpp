#pragma once
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

			void printLayerInfo() override
			{
				printDoubleLine();
				std::cout << "BatchNorm2d Layer" << std::endl;
				std::cout << "	Channel         = " << mDataShape.channel << std::endl;
				std::cout << "	Height          = " << mDataShape.height << std::endl;
				std::cout << "	Width           = " << mDataShape.width << std::endl;
			}

		private:
			static u32 InstanceCounter;
			u32 mInstanceID;

			const f32 ep = 1e-7;

			u32 mBatchSize;
			DataShape mDataShape;

			DataArray mIntermediateResultOnCPU;
			DataArray mSigmaOnCPU;
			
			DataArray mIntermediateResultOnGPU;
			DataArray mSigmaOnGPU;
			DataArray mMeanOnGPU;
			DataArray mBlockSqMeanOnGPU;
			DataArray mBlockMeanOnGPU;

			DataArray mDMeanOnGPU;
			DataArray mDIMeanOnGPU;

			DataArray mBlockDMeanOnGPU;
			DataArray mBlockDIMeanOnGPU;
			DataArray mBlockDGammaOnGPU;
			DataArray mBlockDBetaOnGPU;
		};

	}
}