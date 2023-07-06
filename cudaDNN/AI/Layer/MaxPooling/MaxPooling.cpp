//#include <random>
//#include <cuda_runtime.h>
//#include <iostream>
//#include <cassert>
//#include "MaxPooling.h"
//
//namespace Aoba::layer
//{
//
//	u32 MaxPooling::InstanceCounter = 0;
//
//	MaxPooling::MaxPooling(u32 filterSize, u32 stride, u32 padding)
//		: mBatchSize(0)
//		, mFh(filterSize)
//		, mFw(filterSize)
//		, mSh(stride)
//		, mSw(stride)
//		, mPh(padding)
//		, mPw(padding)
//	{
//	}
//
//	MaxPooling::MaxPooling(
//		u32 filterHeight, u32 filterWidth,
//		u32 strideHeight, u32 strideWidth,
//		u32 paddingHeight, u32 paddingWidth)
//		: mBatchSize(0)
//		, mFh(filterHeight)
//		, mFw(filterWidth)
//		, mSh(strideHeight)
//		, mSw(strideWidth)
//		, mPh(paddingHeight)
//		, mPw(paddingWidth)
//	{
//	}
//
//	MaxPooling::~MaxPooling()
//	{
//	}
//
//	void MaxPooling::setupLayerInfo(u32 batchSize, DataShape& shape)
//	{
//		mBatchSize = batchSize;
//		mInputDataShape = shape;
//
//		mIc = shape.channel;
//		mIh = shape.height;
//		mIw = shape.width;
//
//		mOc = mOutputDataShape.channel = mIc;
//		mOh = mOutputDataShape.height = 1 + (mIh - mFh + 2 * mPh) / mSh;
//		mOw = mOutputDataShape.width = 1 + (mIw - mFw + 2 * mPw) / mSw;
//
//		shape = mOutputDataShape;
//
//
//		mFhFw = mFh * mFw;
//		mIcFhFw = mIc * mFhFw;
//		mIhIw = mIh * mIw;
//		mIcIhIw = mIc * mIhIw;
//		mOhOw = mOh * mOw;
//		mOcOhOw = mOc * mOhOw;
//
//		mInstanceID = InstanceCounter;
//		InstanceCounter++;
//	}
//
//	void MaxPooling::mallocOnCPU()
//	{
//		////////////////////////////////////////////////////////
//		//MaxPoolingの伝搬結果のメモリ確保と初期化
//		////////////////////////////////////////////////////////
//		mForwardResultOnCPU.setSizeAs4D(mBatchSize, mOc, mOh, mOw);
//		//mReshapedInputDataOnCPU.setSizeAs3D(mBatchSize, mOhOw, mIcFhFw);
//		mBackwardResultOnCPU.setSizeAs4D(mBatchSize, mIc, mIh, mIw);
//
//		MALLOC_AND_INITIALIZE_0_ON_CPU(mForwardResultOnCPU);
//		//MALLOC_AND_INITIALIZE_0_ON_CPU(mReshapedInputDataOnCPU);
//		MALLOC_AND_INITIALIZE_0_ON_CPU(mBackwardResultOnCPU);
//
//
//		////////////////////////////////////////////////////////
//		//Maskのメモリ確保と初期化
//		////////////////////////////////////////////////////////
//		mPoolingMaskOnCPU.size = mBatchSize * mOcOhOw;
//		MALLOC_AND_INITIALIZE_0_ON_CPU(mPoolingMaskOnCPU);
//	}
//
//	void MaxPooling::forwardOnCPU()
//	{
//		auto& input = *mInputDataOnCPU;
//
//#if TIME_DEBUG
//		{
//			std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
//#endif
//			for (u32 N = 0; N < mBatchSize; N++)
//			{
//				for (u32 OcOhOw = 0, end = mForwardResultOnCPU.size / mBatchSize; OcOhOw < end; OcOhOw++)
//				{
//					const u32 Oc = OcOhOw / mOhOw;
//					const u32 OhOw = OcOhOw - Oc * mOhOw;
//					const u32 Oh = OhOw / mOw;
//					const u32 Ow = OhOw - mOw * OhOw;
//
//					f32 tmp = 0.0f;
//					u32 maxCandIndex = Oh * mSh * ;
//					f32 maxCandValue = input(N, Oc, mOhOw);
//
//
//					mPoolingMaskOnCPU[N * mOcOhOw + OcOhOw] = maxCandIndex;
//					mForwardResultOnCPU(N, OcOhOw) = maxCandValue;
//				}
//			}
//#if TIME_DEBUG
//			f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
//			std::string name = "";
//			(((name += __FUNCTION__) += " : ") += std::to_string(mInstanceID)) += " : forward";
//			timers[name] = elapsedTime;
//		}
//#endif
//	}
//
//	void MaxPooling::backwardOnCPU()
//	{
//		auto& dout = *mDInputDataOnCPU;
//
//
//#if TIME_DEBUG
//		{
//			std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
//#endif
//			for (u32 c = 0; c < mOc; c++)
//			{
//				//フィルター行列の逆伝搬
//				{
//					for (u32 icfhfw = 0; icfhfw < mIcFhFw; icfhfw++)
//					{
//						f32 tmp = 0;
//
//						for (u32 N = 0; N < mBatchSize; N++)
//						{
//							for (u32 hw = 0; hw < mOhOw; hw++)
//							{
//								tmp += dout(N, c, hw) * mReshapedInputDataOnCPU(N, hw, icfhfw);
//							}
//						}
//						dConvMatrix(c, icfhfw) = tmp;
//					}
//				}
//
//				//バイアスの逆伝搬
//				{
//					f32 tmp = 0.0f;
//					for (u32 N = 0; N < mBatchSize; N++)
//					{
//						for (u32 hw = 0; hw < mOhOw; hw++)
//						{
//							tmp += dout(N, c, hw);
//						}
//					}
//					dConvBias[c] = tmp;
//				}
//			}
//#if TIME_DEBUG
//			f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
//			std::string name = "";
//			(((name += __FUNCTION__) += " : ") += std::to_string(mInstanceID)) += " : backward dF db";
//			timers[name] = elapsedTime;
//		}
//#endif
//
//
//#if TIME_DEBUG
//		{
//			std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
//#endif
//#if 0//レガシーコード
//			for (u32 N = 0; N < mBatchSize; N++)
//			{
//				for (u32 IcIhIw = 0; IcIhIw < mIcIhIw; IcIhIw++)
//				{
//					mBackwardResultOnCPU[N * mIcIhIw + IcIhIw] = 0.0f;
//				}
//
//				for (u32 i = 0, end = mReshapedInputDataOnCPU.size / mBatchSize; i < end; i++)
//				{
//					/*f32 tmp = 0.0f;
//					for (u32 j = 0; j < mOc; j++)
//					{
//						u32 OhOw = i / mIcFhFw;
//						u32 IcFhFw = i - OhOw * mIcFhFw;
//
//						tmp += dout[N * mOcOhOw + j * mOhOw + OhOw] * convMatrix[j * mIcFhFw + IcFhFw];
//					}*/
//
//					u32 OhOw = i / mIcFhFw;
//					u32 fCol = OhOw / mOw;
//					u32 fRow = OhOw - fCol * mOw;
//
//					u32 iRes = i - mIcFhFw * OhOw;
//					u32 c = iRes / mFhFw;
//					u32 h = (iRes - c * mFhFw) / mFw;
//					u32 w = iRes % mFw;
//
//					u32 heightIndex = fCol * mSh + (h - mPh);
//					u32 widthIndex = fRow * mSw + (w - mPw);
//					if (heightIndex < 0 || heightIndex >= mInputDataShape.height || widthIndex < 0 || widthIndex >= mInputDataShape.width)
//					{
//						continue;
//					}
//
//					f32 tmp = 0.0f;
//					for (u32 j = 0; j < mOc; j++)
//					{
//						u32 OhOw = i / mIcFhFw;
//						u32 IcFhFw = i - OhOw * mIcFhFw;
//
//						tmp += dout[N * mOcOhOw + j * mOhOw + OhOw] * convMatrix[j * mIcFhFw + IcFhFw];
//					}
//
//					mBackwardResultOnCPU(N, c, heightIndex, widthIndex) += tmp;
//				}
//			}
//#else
//			for (u32 N = 0; N < mBatchSize; N++)
//			{
//				for (u32 IcIhIw = 0; IcIhIw < mIcIhIw; IcIhIw++)
//				{
//					const u32 c = IcIhIw / mIhIw;
//					const u32 h = (IcIhIw - c * mIhIw) / mIw;
//					const u32 w = IcIhIw % mIw;
//
//					const u32 exH = h + mPh;
//					const u32 exW = w + mPw;
//
//					f32 result = 0.0f;
//					for (u32 Oh = (exH < mFh ? 0 : 1 + (exH - mFh) / mSh), endOh = std::min(1 + (exH / mSh), mOh); Oh < endOh; Oh++)
//					{
//						for (u32 Ow = (exW < mFw ? 0 : 1 + (exW - mFw) / mSw), endOw = std::min(1 + (exW / mSw), mOw); Ow < endOw; Ow++)
//						{
//							const u32 row = Oh * mOw + Ow;
//							const u32 col = c * mFhFw + (exH - Oh * mSh) * mFw + (exW - Ow * mSw);
//							for (u32 Fc = 0; Fc < mOc; Fc++)
//							{
//								result += dout(N, Fc, row) * convMatrix(Fc, col);
//							}
//						}
//					}
//					mBackwardResultOnCPU(N, IcIhIw) = result;
//				}
//			}
//#endif
//#if TIME_DEBUG
//			f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
//			std::string name = "";
//			(((name += __FUNCTION__) += " : ") += std::to_string(mInstanceID)) += " : backward dout";
//			timers[name] = elapsedTime;
//		}
//#endif
//	}
//
//	void MaxPooling::terminateOnCPU()
//	{
//		for (u32 id = 0; id < mParametersPtrOnCPU.size(); id++)
//		{
//			delete[] mParametersPtrOnCPU[id].address;
//			delete[] mDParametersPtrOnCPU[id].address;
//		}
//
//		delete[] mForwardResultOnCPU.address;
//		delete[] mReshapedInputDataOnCPU.address;
//		delete[] mBackwardResultOnCPU.address;
//	}
//}