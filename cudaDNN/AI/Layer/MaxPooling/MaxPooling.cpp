#include <random>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include "MaxPooling.h"

namespace Aoba::layer
{

	u32 MaxPooling::InstanceCounter = 0;

	MaxPooling::MaxPooling(u32 filterSize, u32 stride, u32 padding)
		: mBatchSize(0)
		, mFh(filterSize)
		, mFw(filterSize)
		, mSh(stride)
		, mSw(stride)
		, mPh(padding)
		, mPw(padding)
	{
	}

	MaxPooling::MaxPooling(
		u32 filterHeight, u32 filterWidth,
		u32 strideHeight, u32 strideWidth,
		u32 paddingHeight, u32 paddingWidth)
		: mBatchSize(0)
		, mFh(filterHeight)
		, mFw(filterWidth)
		, mSh(strideHeight)
		, mSw(strideWidth)
		, mPh(paddingHeight)
		, mPw(paddingWidth)
	{
	}

	MaxPooling::~MaxPooling()
	{
	}

	void MaxPooling::setupLayerInfo(u32 batchSize, DataShape& shape)
	{
		mBatchSize = batchSize;
		mInputDataShape = shape;

		mIc = shape.channel;
		mIh = shape.height;
		mIw = shape.width;

		mOc = mOutputDataShape.channel = mIc;
		mOh = mOutputDataShape.height = 1 + (mIh - mFh + 2 * mPh) / mSh;
		mOw = mOutputDataShape.width = 1 + (mIw - mFw + 2 * mPw) / mSw;

		shape = mOutputDataShape;


		mFhFw = mFh * mFw;
		mIcFhFw = mIc * mFhFw;
		mIhIw = mIh * mIw;
		mIcIhIw = mIc * mIhIw;
		mOhOw = mOh * mOw;
		mOcOhOw = mOc * mOhOw;

		mInstanceID = InstanceCounter;
		InstanceCounter++;
	}

	void MaxPooling::mallocOnCPU()
	{
		////////////////////////////////////////////////////////
		//MaxPoolingの伝搬結果のメモリ確保と初期化
		////////////////////////////////////////////////////////
		mForwardResultOnCPU.setSizeAs4D(mBatchSize, mOc, mOh, mOw);
		//mReshapedInputDataOnCPU.setSizeAs3D(mBatchSize, mOhOw, mIcFhFw);
		mBackwardResultOnCPU.setSizeAs4D(mBatchSize, mIc, mIh, mIw);

		MALLOC_AND_INITIALIZE_0_ON_CPU(mForwardResultOnCPU);
		//MALLOC_AND_INITIALIZE_0_ON_CPU(mReshapedInputDataOnCPU);
		MALLOC_AND_INITIALIZE_0_ON_CPU(mBackwardResultOnCPU);


		////////////////////////////////////////////////////////
		//Maskのメモリ確保と初期化
		////////////////////////////////////////////////////////
		mPoolingMaskOnCPU.resize(mBatchSize * mOcOhOw);
		for (auto& comp : mPoolingMaskOnCPU)
		{
			comp = 0;
		}
	}

	void MaxPooling::forwardOnCPU()
	{
		auto& input = *mInputDataOnCPU;

#if TIME_DEBUG
		{
			std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
			auto rangeChecker = [](s32 index, u32 maxIndex) -> bool {return (0 <= index && index < maxIndex);};
			auto rangeCheckerHW = [&rangeChecker](s32 indexH, s32 indexW, u32 maxIndexH, u32 maxIndexW) -> bool {return rangeChecker(indexH, maxIndexH) && rangeChecker(indexW, maxIndexW);};

			for (u32 N = 0; N < mBatchSize; N++)
			{
				for (u32 OcOhOw = 0; OcOhOw < mOcOhOw; OcOhOw++)
				{
					const u32 Oc = OcOhOw / mOhOw;
					const u32 Ic = Oc;
					const u32 OhOw = OcOhOw - Oc * mOhOw;
					const u32 Oh = OhOw / mOw;
					const u32 Ow = OhOw - Oh * mOw;

					const s32 basisIndexIh = Oh * mSh - mPh;
					const s32 basisIndexIw = Ow * mSw - mPw;

					s32 maxCandIh = basisIndexIh;
					s32 maxCandIw = basisIndexIw;
					f32 maxValueCand = rangeCheckerHW(maxCandIh,maxCandIw, mIh, mIw) ? input(N, Ic, maxCandIh, maxCandIw) : 0;

					for (u32 fh = 0; fh < mFh; fh++)
					{
						for (u32 fw = 0; fw < mFw; fw++)
						{
							const s32 indexIh = basisIndexIh + fh;
							const s32 indexIw = basisIndexIw + fw;

							const f32 value = rangeCheckerHW(indexIh, indexIw, mIh, mIw) ?
								input(N, Ic, indexIh, indexIw) : 0;

							if (maxValueCand < value)
							{
								maxValueCand = value;
								maxCandIh = indexIh;
								maxCandIw = indexIw;
							}
						}
					}

					mPoolingMaskOnCPU[N * mOcOhOw + OcOhOw] = 
						rangeCheckerHW(maxCandIh, maxCandIw, mIh, mIw) ?
						Ic * mIhIw + maxCandIh * mIw + maxCandIw : -1;
					mForwardResultOnCPU(N, OcOhOw) = maxValueCand;
				}
			}
#if TIME_DEBUG
			f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
			std::string name = "";
			(((name += __FUNCTION__) += " : ") += std::to_string(mInstanceID)) += " : forward";
			timers[name] = elapsedTime;
		}
#endif
	}

	void MaxPooling::backwardOnCPU()
	{
		auto& dout = *mDInputDataOnCPU;

		//逆伝搬用の出力を全て0にリセット
#if TIME_DEBUG
		{
			std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif

			for (u32 N = 0; N < mBatchSize; N++)
			{
				for (u32 IcIhIw = 0; IcIhIw < mIcIhIw; IcIhIw++)
				{
					mBackwardResultOnCPU(N, IcIhIw) = 0;
				}
			}

#if TIME_DEBUG
			f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
			std::string name = "";
			(((name += __FUNCTION__) += " : ") += std::to_string(mInstanceID)) += " : backward init";
			timers[name] = elapsedTime;
		}
#endif

#if TIME_DEBUG
		{
			std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif

			for (u32 N = 0; N < mBatchSize; N++)
			{
				for (u32 OcOhOw = 0; OcOhOw < mOcOhOw; OcOhOw++)
				{
					s32 index = mPoolingMaskOnCPU[N * mOcOhOw + OcOhOw];
					if (index < 0)
					{
						continue;
					}

					mBackwardResultOnCPU(N, index) += dout[N * mOcOhOw + OcOhOw];
				}
			}

#if TIME_DEBUG
			f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
			std::string name = "";
			(((name += __FUNCTION__) += " : ") += std::to_string(mInstanceID)) += " : backward";
			timers[name] = elapsedTime;
		}
#endif
	}

	void MaxPooling::terminateOnCPU()
	{
		delete[] mForwardResultOnCPU.address;
		delete[] mBackwardResultOnCPU.address;
	}
}