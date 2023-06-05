#pragma once
#include "../../../common.h"
#include "../../AISetting.h"
#include "../BaseLayer.h"

namespace Aoba
{
	namespace layer
	{
		class Convolution : public BaseLayer
		{
		public:
			Convolution(u32, u32, u32, f32 weight = 0.01f);
			Convolution(u32, u32, u32, u32, f32 weight = 0.01f);
			Convolution(u32, u32, u32, u32, u32, f32 weight = 0.01f);
			~Convolution();

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
				std::cout << "Convolution Layer" << std::endl;
				std::cout << "	InputSize              = " << "(" << mInputDataShape.channel << " , " << mInputDataShape.height << " , " << mInputDataShape.width << ")" << std::endl;
				std::cout << "	OutputSize             = " << "(" << mOutputDataShape.channel << " , " << mOutputDataShape.height << " , " << mOutputDataShape.width << ")" << std::endl;
				std::cout << "	KernelSize             = " << "(" << mFilterHeight << " , " << mFilterWidth << ")" << std::endl;
				if (mStrideHeight == mStrideWidth)
				{
					std::cout << "	Stride                 = " << mStrideHeight << std::endl;
				}
				else
				{
					std::cout << "	Stride                 = " << "(" << mStrideHeight << "," << mStrideWidth << ")" << std::endl;
				}
				if (mPaddingHeight == mPaddingWidth)
				{
					std::cout << "	Padding                = " << mPaddingHeight << std::endl;
				}
				else
				{
					std::cout << "	Padding                = " << "(" << mPaddingHeight << "," << mPaddingWidth << ")" << std::endl;
				}
				std::cout << "	ConvolutionParamWeight = " << mConvolutionParamWeight << std::endl;
			}


		private:

			u32 mBatchSize;
			DataShape mInputDataShape;
			DataShape mOutputDataShape;
			u32 mFilterHeight;
			u32 mFilterWidth;
			u32 mFilterNum;
			u32 mStrideHeight;
			u32 mStrideWidth;
			u32 mPaddingHeight;
			u32 mPaddingWidth;

			u32 mIc;
			u32 mIh;
			u32 mIw;
			u32 mOc;
			u32 mOh;
			u32 mOw;
			u32 mFhFw;
			u32 mIcFhFw;
			u32 mIhIw;
			u32 mIcIhIw;
			u32 mOhOw;
			u32 mOcOhOw;

			f32 mConvolutionParamWeight = 0.01f;


			DataArray mReshapedInputDataOnCPU;
			DataArray mReshapedInputDataOnGPU;

		};

	}
}