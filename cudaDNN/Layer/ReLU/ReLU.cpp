#include <iostream>
#include <cassert>

#include "../../commonCPU.h"
#include "ReLU.h"


namespace miduho::layer
{
	ReLU::ReLU()
		:mBatchSize(0)
		, mInputSize(0)
		, mOutputSize(0)
	{
	}

	ReLU::~ReLU(){}


	void ReLU::setupLayerInfo(FlowDataFormat* pInputData)
	{
		mBatchSize = pInputData->batchSize;
		mInputSize = pInputData->width;
		mOutputSize = mInputSize;

		isInitialized = true;
	}

	void ReLU::initialize()
	{
		assert(isInitialized);

#ifdef GPU_AVAILABLE
		initializeOnGPU();
#else
		initializeOnCPU();
#endif // GPUA_VAILABLE
	}

	void ReLU::forward()
	{
#ifdef GPU_AVAILABLE
		forwardOnGPU();
#else
		forwardOnCPU();
#endif	
	}

	void ReLU::backward()
	{
#ifdef GPU_AVAILABLE
		backwardOnGPU();
#else
		backwardOnCPU();
#endif	
	}

	void ReLU::terminate()
	{
#ifdef GPU_AVAILABLE
		terminateOnGPU();
#else
		terminateOnCPU();
#endif // GPUA_VAILABLE
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
