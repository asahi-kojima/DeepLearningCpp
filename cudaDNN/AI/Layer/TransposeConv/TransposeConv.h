//#pragma once
//#include "../BaseLayer.h"
//
//namespace Aoba
//{
//	namespace layer
//	{
//		class TransposeConv : public BaseLayer
//		{
//		public:
//			TransposeConv(u32, u32, u32, f32 weight = 0.01f);
//			TransposeConv(u32, u32, u32, u32, f32 weight = 0.01f);
//			TransposeConv(u32, u32, u32, u32, u32, u32, u32, f32 weight = 0.01f);
//			~TransposeConv();
//
//		private:
//			void setupLayerInfo(u32, DataShape&) override;
//
//
//			void mallocOnCPU() override;
//			void forwardOnCPU()  override;
//			void backwardOnCPU() override;
//			void terminateOnCPU() override;
//
//			void mallocOnGPU() override;
//			void forwardOnGPU()  override;
//			void backwardOnGPU() override;
//			void terminateOnGPU() override;
//
//			void printLayerInfo() override
//			{
//				printDoubleLine();
//				printLayerName("TransposeConv Layer");
//				print3dProperty("InputSize", mInputDataShape);
//				print3dProperty("OutputSize", mOutputDataShape);
//				print2dProperty("KernelSize", mFilterHeight, mFilterWidth);
//				if (mStrideHeight == mStrideWidth)
//				{
//					printProperty("Stride", mStrideHeight);
//				}
//				else
//				{
//					print2dProperty("Stride", mStrideHeight, mStrideWidth);
//				}
//				if (mPaddingHeight == mPaddingWidth)
//				{
//					printProperty("Padding", mPaddingHeight);
//				}
//				else
//				{
//					print2dProperty("Padding", mPaddingHeight, mPaddingWidth);
//				}
//				printProperty("ParamWeight", mTransposeConvParamWeight);
//			}
//
//
//		private:
//			static u32 InstanceCounter;
//			u32 mInstanceID;
//			u32 mBatchSize;
//			DataShape mInputDataShape;
//			DataShape mOutputDataShape;
//			u32 mFilterHeight;
//			u32 mFilterWidth;
//			u32 mFilterNum;
//			u32 mStrideHeight;
//			u32 mStrideWidth;
//			u32 mPaddingHeight;
//			u32 mPaddingWidth;
//
//			u32 mIc;
//			u32 mIh;
//			u32 mIw;
//			u32 mOc;
//			u32 mOh;
//			u32 mOw;
//			u32 mFh;
//			u32 mFw;
//			u32 mSh;
//			u32 mSw;
//			u32 mFhFw;
//			u32 mIcFhFw;
//			u32 mIhIw;
//			u32 mIcIhIw;
//			u32 mOhOw;
//			u32 mOcOhOw;
//
//			f32 mTransposeConvParamWeight = 0.01f;
//
//
//			DataArray mReshapedInputDataOnCPU;
//			DataArray mReshapedInputDataOnGPU;
//		public:
//			struct parameterInfo
//			{
//				u32 batchSize;
//				u32 Ih;
//				u32 Iw;
//				u32 Ic;
//				u32 IhIw;
//				u32 IcIhIw;
//				u32 IcFhFw;
//				u32 Oc;
//				u32 Oh;
//				u32 Ow;
//				u32 OhOw;
//				u32 OcOhOw;
//				u32 OhOwIcFhFw;
//				u32 Fn;
//				u32 Fh;
//				u32 Fw;
//				u32 FhFw;
//				u32 Sh;
//				u32 Sw;
//				u32 Ph;
//				u32 Pw;
//			};
//		private:
//			parameterInfo* mParameterInfoOnGPU;
//		};
//
//	}
//}