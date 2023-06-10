#include <random>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

#include "Convolution.h"
#include "../../../common.h"

namespace Aoba::layer
{
	namespace
	{
		void filterInfoCalculator(u32 outputSize, u32 inputSize, u32& filterSize, u32& stride, u32& padding)
		{
			u32 S = 1;
			while (1)
			{
				if ((S + 1) * (outputSize - 1) <= inputSize)
				{
					S++;
				}
				else
				{
					break;
				}
			}
		};
	}



	Convolution::Convolution(u32 outputChannel, u32 outputHeight, u32 outputWidth, f32 weight)
		: mBatchSize(0)
		, mOutputDataShape{ outputChannel, outputHeight, outputWidth }
		, mConvolutionParamWeight(weight)
	{
		mIcFhFw = mInputDataShape.channel * mFilterHeight * mFilterWidth;
		mIhIw = mInputDataShape.height * mInputDataShape.width;
		mIcIhIw = mInputDataShape.channel * mIhIw;
		mOhOw = mOutputDataShape.height * mOutputDataShape.width;
		mOcOhOw = mOutputDataShape.channel * mOhOw;

		mFilterNum = mOutputDataShape.channel;
	}

	Convolution::Convolution(u32 filterNum, u32 filterSize, u32 stride, u32 padding, f32 weight)
		: mBatchSize(0)
		, mFilterNum(filterNum)
		, mFilterHeight(filterSize)
		, mFilterWidth(filterSize)
		, mStrideHeight(stride)
		, mStrideWidth(stride)
		, mPaddingHeight(padding)
		, mPaddingWidth(padding)
		, mConvolutionParamWeight(weight)
	{
		mOutputDataShape.channel = mFilterNum;
		mOutputDataShape.height = 1 + (mInputDataShape.height - mFilterHeight + 2 * mPaddingHeight) / mStrideHeight;
		mOutputDataShape.width = 1 + (mInputDataShape.width - mFilterWidth + 2 * mPaddingWidth) / mStrideWidth;
		mIcFhFw = mInputDataShape.channel * mFilterHeight * mFilterWidth;
		mIhIw = mInputDataShape.height * mInputDataShape.width;
		mIcIhIw = mInputDataShape.channel * mIhIw;
		mOhOw = mOutputDataShape.height * mOutputDataShape.width;
		mOcOhOw = mOutputDataShape.channel * mOhOw;
	}

	Convolution::Convolution(u32 filterNum, u32 filterHeight, u32 filterWidth, u32 strideHeight, u32 strideWidth, u32 paddingHeight, u32 paddingWidth, f32 weight)
		: mBatchSize(0)
		, mFilterNum(filterNum)
		, mFilterHeight(filterHeight)
		, mFilterWidth(filterWidth)
		, mStrideHeight(strideHeight)
		, mStrideWidth(strideWidth)
		, mPaddingHeight(paddingHeight)
		, mPaddingWidth(paddingWidth)
		, mConvolutionParamWeight(weight)
	{
	}

	Convolution::~Convolution()
	{
	}

	void Convolution::setupLayerInfo(u32 batchSize, DataShape& shape)
	{
		mBatchSize = batchSize;
		mInputDataShape = shape;

		mIc = shape.channel;
		mIh = shape.height;
		mIw = shape.width;

		mOc = mOutputDataShape.channel = mFilterNum;
		mOh = mOutputDataShape.height = 1 + (mIh - mFilterHeight + 2 * mPaddingHeight) / mStrideHeight;
		mOw = mOutputDataShape.width = 1 + (mIw - mFilterWidth + 2 * mPaddingWidth) / mStrideWidth;
		shape = mOutputDataShape;


		mFhFw = mFilterHeight * mFilterWidth;
		mIcFhFw = mIc * mFhFw;
		mIhIw = mIh * mIw;
		mIcIhIw = mIc * mIhIw;
		mOhOw = mOh * mOw;
		mOcOhOw = mOc * mOhOw;
	}

	void Convolution::mallocOnCPU()
	{
		mParametersPtrOnCPU.resize(2);
		mDParametersPtrOnCPU.resize(2);

		////////////////////////////////////////////////////////
		//Convolutionのフィルターパラメータのメモリ確保と初期化
		////////////////////////////////////////////////////////
		DataArray& convParam = mParametersPtrOnCPU[0];
		DataArray& convDParam = mDParametersPtrOnCPU[0];
		convParam.setSizeAs2D(mFilterNum, mIcFhFw);
		convDParam.setSizeAs2D(mFilterNum, mIcFhFw);
		MALLOC_AND_INITIALIZE_NORMAL_ON_CPU(convParam, mIcFhFw, mConvolutionParamWeight);
		MALLOC_AND_INITIALIZE_0_ON_CPU(convDParam);


		////////////////////////////////////////////////////////
		//Convolutionのbaisパラメータのメモリ確保と初期化
		////////////////////////////////////////////////////////
		DataArray& biasParam = mParametersPtrOnCPU[1];
		DataArray& biasDParam = mDParametersPtrOnCPU[1];
		biasParam.size = biasDParam.size = mOutputDataShape.channel;
		MALLOC_AND_INITIALIZE_0_ON_CPU(biasParam);
		MALLOC_AND_INITIALIZE_0_ON_CPU(biasDParam);

		////////////////////////////////////////////////////////
		//Convolutionの伝搬結果のメモリ確保と初期化
		////////////////////////////////////////////////////////
		mForwardResultOnCPU.setSizeAs4D(mBatchSize, mOc, mOh, mOw);
		mReshapedInputDataOnCPU.setSizeAs3D(mBatchSize, mOhOw, mIcFhFw);
		mBackwardResultOnCPU.setSizeAs4D(mBatchSize, mIc, mIh, mIw);

		MALLOC_AND_INITIALIZE_0_ON_CPU(mForwardResultOnCPU);
		MALLOC_AND_INITIALIZE_0_ON_CPU(mReshapedInputDataOnCPU);
		MALLOC_AND_INITIALIZE_0_ON_CPU(mBackwardResultOnCPU);
	}

	void Convolution::forwardOnCPU()
	{
		auto& input = *mInputDataOnCPU;
		DataArray& convMatrix = mParametersPtrOnCPU[0];
		DataArray& convBias = mParametersPtrOnCPU[1];

#if TIME_DEBUG
		{
			std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
			for (u32 N = 0; N < mBatchSize; N++)
			{
				for (u32 i = 0, end = mReshapedInputDataOnCPU.size / mBatchSize; i < end; i++)
				{
					u32 V = i / mIcFhFw;
					u32 H = i - V * mIcFhFw;

					u32 Fh = V / mOw;
					u32 Fw = V - Fh * mOw;

					u32 c = H / mFhFw;
					u32 h = (H - c * mFhFw) / mFilterWidth;
					u32 w = H % mFilterWidth;

					u32 indexH = Fh * mStrideHeight + (h - mPaddingHeight);
					u32 indexW = Fw * mStrideWidth + (w - mPaddingWidth);
					bool isValid = !(indexH < 0 || indexH >= mIh || indexW < 0 || indexW >= mIw);
					mReshapedInputDataOnCPU(N, i) =
						isValid ?
						input(N, c, indexH, indexW) :
						//input(N, c, mFilterHeight * mStrideHeight + (h - mPaddingHeight), mFilterWidth * mStrideWidth + (w - mPaddingWidth)) :
						0.0f;
				}

				for (u32 OcOhOw = 0, end = mForwardResultOnCPU.size / mBatchSize; OcOhOw < end; OcOhOw++)
				{
					f32 tmp = 0.0f;
					const u32 Fc = OcOhOw / mOhOw;
					const u32 OhOw = OcOhOw - Fc * mOhOw;

					for (u32 IcFhFw = 0; IcFhFw < mIcFhFw; IcFhFw++)
					{
						tmp += convMatrix(Fc, IcFhFw) * mReshapedInputDataOnCPU(N, OhOw, IcFhFw);
					}
					mForwardResultOnCPU(N, OcOhOw) = tmp + convBias[Fc];
				}
			}

#if TIME_DEBUG
			f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
			std::string name = "";
			(name += __FUNCTION__) += " : forward";
			timers[name] = elapsedTime;
		}
#endif
	}

	void Convolution::backwardOnCPU()
	{
		auto& dout = *mDInputDataOnCPU;

		DataArray& convMatrix = mParametersPtrOnCPU[0];
		DataArray& dConvMatrix = mDParametersPtrOnCPU[0];
		DataArray& convBias = mParametersPtrOnCPU[1];
		DataArray& dConvBias = mDParametersPtrOnCPU[1];


		//for (u32 i = 0; i < mOc * mIcFhFw; i++)
		//{
		//	f32 tmp = 0;
		//	u32 c = i / mIcFhFw;
		//	u32 icfhfw = i - mIcFhFw * c;
		//	for (u32 N = 0; N < mBatchSize; N++)
		//	{
		//		for (u32 hw = 0; hw < mOhOw; hw++)
		//		{
		//			tmp += dout[N * mOcOhOw + c * mOhOw + hw] * mReshapedInputDataOnCPU(N, hw, icfhfw);
		//		}
		//	}
		//	dConvMatrix[i] = tmp;
		//}
		//for (u32 c = 0; c < mOc; c++)
		//{
		//	f32 tmp = 0.0f;
		//	for (u32 N = 0; N < mBatchSize; N++)
		//	{
		//		for (u32 hw = 0; hw < mOhOw; hw++)
		//		{
		//			tmp += dout(N, c, hw);
		//		}
		//	}
		//	dConvBias[c] = tmp;
		//}
#if TIME_DEBUG
		{
			std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
			for (u32 c = 0; c < mOc; c++)
			{
				//フィルター行列の逆伝搬
				{
					for (u32 icfhfw = 0; icfhfw < mIcFhFw; icfhfw++)
					{
						f32 tmp = 0;

						for (u32 N = 0; N < mBatchSize; N++)
						{
							for (u32 hw = 0; hw < mOhOw; hw++)
							{
								tmp += dout(N, c, hw) * mReshapedInputDataOnCPU(N, hw, icfhfw);
							}
						}
						dConvMatrix(c, icfhfw) = tmp;
					}
				}

				//バイアスの逆伝搬
				{
					f32 tmp = 0.0f;
					for (u32 N = 0; N < mBatchSize; N++)
					{
						for (u32 hw = 0; hw < mOhOw; hw++)
						{
							tmp += dout(N, c, hw);
						}
					}
					dConvBias[c] = tmp;
				}
			}
#if TIME_DEBUG
			f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
			std::string name = "";
			(name += __FUNCTION__) += " : backward dF db";
			timers[name] = elapsedTime;
		}
#endif


#if TIME_DEBUG
		{
			std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
			for (u32 N = 0; N < mBatchSize; N++)
			{
				for (u32 IcIhIw = 0; IcIhIw < mIcIhIw; IcIhIw++)
				{
					mBackwardResultOnCPU[N * mIcIhIw + IcIhIw] = 0.0f;
				}

				for (u32 i = 0, end = mReshapedInputDataOnCPU.size / mBatchSize; i < end; i++)
				{
					f32 tmp = 0.0f;
					for (u32 j = 0; j < mFilterNum; j++)
					{
						u32 OhOw = i / mIcFhFw;
						u32 IcFhFw = i - OhOw * mIcFhFw;

						tmp += dout[N * mOcOhOw + j * mOhOw + OhOw] * convMatrix[j * mIcFhFw + IcFhFw];
					}

					u32 OhOw = i / mIcFhFw;
					u32 fCol = OhOw / mOw;
					u32 fRow = OhOw - fCol * mOw;

					u32 iRes = i - mIcFhFw * OhOw;
					u32 c = iRes / mFhFw;
					u32 h = (iRes - c * mFhFw) / mFilterWidth;
					u32 w = iRes % mFilterWidth;

					u32 heightIndex = fCol * mStrideHeight + (h - mPaddingHeight);
					u32 widthIndex = fRow * mStrideWidth + (w - mPaddingWidth);
					if (heightIndex < 0 || heightIndex >= mInputDataShape.height || widthIndex < 0 || widthIndex >= mInputDataShape.width)
					{
						continue;
					}
					mBackwardResultOnCPU(N, c, heightIndex, widthIndex) += tmp;
				}
			}
#if TIME_DEBUG
			f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
			std::string name = "";
			(name += __FUNCTION__) += " : backward dout";
			timers[name] = elapsedTime;
		}
#endif
	}

	void Convolution::terminateOnCPU()
	{

	}
}