#pragma once
#include "../BaseLayer.h"

namespace Aoba
{
	namespace layer
	{
		class MaxPooling : public BaseLayer
		{
		public:
			//MaxPooling(u32, u32, u32, f32 weight = 0.01f);
			MaxPooling(u32, u32, u32);
			MaxPooling(u32, u32, u32, u32, u32, u32);
			~MaxPooling();

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
				printLayerName("MaxPooling Layer");
				print3dProperty("InputSize", mInputDataShape);
				print3dProperty("OutputSize", mOutputDataShape);
				print2dProperty("KernelSize", mFh, mFw);
				if (mSh == mSw)
				{
					printProperty("Stride", mSh);
				}
				else
				{
					print2dProperty("Stride", mSh, mSw);
				}
				if (mPh == mPw)
				{
					printProperty("Padding", mPh);
				}
				else
				{
					print2dProperty("Padding", mPh, mPw);
				}
			}


		private:
			static u32 InstanceCounter;
			u32 mInstanceID;
			u32 mBatchSize;
			DataShape mInputDataShape;
			DataShape mOutputDataShape;

			u32 mIc;
			u32 mIh;
			u32 mIw;
			u32 mOc;
			u32 mOh;
			u32 mOw;
			u32 mFh;
			u32 mFw;
			u32 mSh;
			u32 mSw;
			u32 mPh;
			u32 mPw;
			u32 mFhFw;
			u32 mIcFhFw;
			u32 mIhIw;
			u32 mIcIhIw;
			u32 mOhOw;
			u32 mOcOhOw;


			std::vector<s32> mPoolingMaskOnCPU;
			s32* mPoolingMaskOnGPU;

		public:
			struct parameterInfo
			{
				u32 batchSize;
				u32 Ih;//
				u32 Iw;//
				u32 Ic;
				u32 IhIw;
				u32 IcIhIw;//
				u32 IcFhFw;
				u32 Oc;
				u32 Oh;
				u32 Ow;//
				u32 OhOw;//
				u32 OcOhOw;
				u32 Fn;
				u32 Fh;//
				u32 Fw;//
				u32 FhFw;
				u32 Sh;//
				u32 Sw;//
				u32 Ph;//
				u32 Pw;//
			};
		private:
			parameterInfo* mParameterInfoOnGPU;
		};

	}
}