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
	}

	void ReLU::backwardOnCPU()
	{
		for (u32 N = 0; N < mBatchSize; N++)
		{
			for (u32 i = 0; i < mDataSize; i++)
			{
				u32 index = N * mDataSize + i;
				mBackwardResultOnCPU.address[index] = mMaskOnCPU[index] * (*mDInputDataOnCPU)[index];
			}
		}
	}

	void ReLU::terminateOnCPU()
	{
		delete[] mMaskOnCPU.address;

		delete[] mForwardResultOnCPU.address ;
		delete[] mBackwardResultOnCPU.address;
	}
}
