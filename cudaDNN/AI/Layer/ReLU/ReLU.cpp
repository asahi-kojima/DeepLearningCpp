#include <iostream>
#include <cassert>

#include "../../../common.h"
#include "ReLU.h"


namespace Aoba::layer
{
	ReLU::ReLU()
		:mBatchSize(0)
		, mDataSize(0)
	{
	}

	ReLU::~ReLU(){}


	void ReLU::setupLayerInfo(u32 batchSize, DataShape& shape)
	{
		mBatchSize = batchSize;
		mDataShape = shape;
		mDataSize = mDataShape.getDataSize();
	}



	//////////////////////////////////////
	//CPU ä÷êî
	//////////////////////////////////////
	void ReLU::mallocOnCPU()
	{
		mMaskOnCPU.size = mBatchSize * mDataSize;
		mallocCPUData(mMaskOnCPU);
		initializeDataOnCPU_1(mMaskOnCPU);


		mForwardResultOnCPU.setSizeAs4D(mBatchSize, mDataShape);
		mBackwardResultOnCPU.setSizeAs4D(mBatchSize, mDataShape);
		mallocCPUData(mForwardResultOnCPU);
		mallocCPUData(mBackwardResultOnCPU);
		initializeDataOnCPU_0(mForwardResultOnCPU);
		initializeDataOnCPU_0(mBackwardResultOnCPU);
	}

	void ReLU::forwardOnCPU()
	{
#if TIME_DEBUG
		{
			std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
		for (u32 N = 0; N < mBatchSize; N++)
		{
			for (u32 i = 0; i < mDataSize; i++)
			{
				u32 index = N * mDataSize + i;
				f32 input = mInputDataOnCPU->address[index];
				if (input > 0)
				{
					mMaskOnCPU.address[index] = 1.0f;
					mForwardResultOnCPU[index] = input;
				}
				else
				{
					mMaskOnCPU.address[index] = 1.0f;
					mForwardResultOnCPU[index] = 0.0f;
				}
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

	void ReLU::backwardOnCPU()
	{
#if TIME_DEBUG
		{
			std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
		for (u32 N = 0; N < mBatchSize; N++)
		{
			for (u32 i = 0; i < mDataSize; i++)
			{
				u32 index = N * mDataSize + i;
				mBackwardResultOnCPU.address[index] = mMaskOnCPU[index] * (*mDInputDataOnCPU)[index];
			}
		}
#if TIME_DEBUG
		f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
		std::string name = "";
		(name += __FUNCTION__) += " : backward";
		timers[name] = elapsedTime;
	}
#endif
	}

	void ReLU::terminateOnCPU()
	{
		delete[] mMaskOnCPU.address;

		delete[] mForwardResultOnCPU.address ;
		delete[] mBackwardResultOnCPU.address;
	}
}
