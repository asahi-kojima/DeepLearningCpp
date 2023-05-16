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
				std::cout << "ReLU Layer" << std::endl;
				std::cout << "	InputSize         = " << mInputSize << std::endl;
				std::cout << "	OutputSize        = " << mOutputSize << std::endl;
			}

		private:
			u32 mOutputSize;
			u32 mInputSize;
			u32 mBatchSize;

			DataMemory mMaskOnCPU;
			DataMemory mMaskOnGPU;
		};

	}
}