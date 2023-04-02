#include <iostream>
#include <cassert>

#include "../../../commonCPU.h"
#include "ReLU.h"


namespace Aoba::layer
{
	ReLU::ReLU()
		:mBatchSize(0)
		, mInputSize(0)
		, mOutputSize(0)
	{
	}

	ReLU::~ReLU(){}


	void ReLU::setupLayerInfo(DataShape* pInputData)
	{
		mBatchSize = pInputData->batchSize;
		mInputSize = pInputData->width;
		mOutputSize = mInputSize;
	}



	//////////////////////////////////////
	//CPU ä÷êî
	//////////////////////////////////////
	void ReLU::initializeOnCPU()
	{
		mMaskOnCPU.size = mBatchSize * mInputSize;;
		mMaskOnCPU.address = new f32[mMaskOnCPU.size];
		for (u32 i = 0; i < mMaskOnCPU.size; i++)
		{
			mMaskOnCPU.address[i] = 1.0f;
		}


		mForwardResultOnCPU.size = mBatchSize * mOutputSize;
		mBackwardResultOnCPU.size = mBatchSize * mInputSize;

		mForwardResultOnCPU.address = new f32[mForwardResultOnCPU.size];
		mBackwardResultOnCPU.address = new f32[mBackwardResultOnCPU.size];

		for (u32 i = 0; i < mForwardResultOnCPU.size; i++)
		{
			mForwardResultOnCPU.address[i] = 1.0f;
			mBackwardResultOnCPU.address[i] = 1.0f;
		}
	}

	void ReLU::forwardOnCPU()
	{
		for (u32 N = 0; N < mBatchSize; N++)
		{
			for (u32 i = 0; i < mInputSize; i++)
			{
				u32 index = N * mInputSize + i;
				f32 input = mInputDataOnCPU->address[index];
				if (input > 0)
				{
					mMaskOnCPU.address[index] = 1.0f;
					mForwardResultOnCPU.address[index] = input;
				}
				else
				{
					mMaskOnCPU.address[index] = 1.0f;
					mForwardResultOnCPU.address[index] = 0.0f;
				}
			}
		}
	}

	void ReLU::backwardOnCPU()
	{
		for (u32 N = 0; N < mBatchSize; N++)
		{
			for (u32 i = 0; i < mInputSize; i++)
			{
				u32 index = N * mInputSize + i;
				mBackwardResultOnCPU.address[index] = mMaskOnCPU.address[index] * mDInputDataOnCPU->address[index];
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
