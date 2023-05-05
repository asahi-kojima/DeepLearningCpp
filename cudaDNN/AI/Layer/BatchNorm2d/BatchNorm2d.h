//#pragma once
//
//#include "../../../setting.h"
//#include "../BaseLayer.h"
//
//namespace Aoba
//{
//	namespace layer
//	{
//		class BatchNorm2d : public BaseLayer
//		{
//		public:
//			BatchNorm2d();
//			~BatchNorm2d();
//
//		private:
//			void setupLayerInfo(dataFormat) override;
//
//
//			void initializeOnCPU() override;
//			void forwardOnCPU()  override;
//			void backwardOnCPU() override;
//			void terminateOnCPU() override;
//
//			void initializeOnGPU() override;
//			void forwardOnGPU()  override;
//			void backwardOnGPU() override;
//			void terminateOnGPU() override;
//
//
//		private:
//			u32 mOutputSize;
//			u32 mInputSize;
//			u32 mBatchSize;
//			u32 mChannel;
//
//		};
//	}
//}