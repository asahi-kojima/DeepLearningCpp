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

		mAlreadySetup = true;
	}


	void ReLU::memcpyHostToDevice()
	{

	}

	void ReLU::memcpyDeviceToHost()
	{

	}


	//////////////////////////////////////
	//CPU ä÷êî
	//////////////////////////////////////
	void ReLU::initializeOnCPU()
	{
		
	}

	void ReLU::forwardOnCPU()
	{

	}

	void ReLU::backwardOnCPU()
	{

	}

	void ReLU::terminateOnCPU()
	{

	}
}
