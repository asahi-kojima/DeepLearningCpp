#include <iostream>
#include <thread>

#include "Layer.h"
#include "Machine.h"
#include "commonGPU.cuh"
#include "commonCPU.h"

#define CREATELAYER(classname, ...) std::make_unique<classname>(__VA_ARGS__)


namespace miduho
{
	u32 Machine::entry()
	{
#ifdef GPUA_VAILABLE
		printLine();
		std::cout << "GPU Information" << std::endl;
		printGpuDriverInfo();
		printLine();
#endif // GPUA_VAILABLE


		bool isInitializeSuccess = initialize();
		if (!isInitializeSuccess)
		{
			return 1;
		}

		bool isPreProcessSuccess = preProcess();
		if (!isPreProcessSuccess)
		{
			return 1;
		}

		bool isMainProcessSuccess = mainProcess();
		if (!isMainProcessSuccess)
		{
			return 1;
		}

		bool isPostProcessSuccess = postProcess();
		if (!isPostProcessSuccess)
		{
			return 1;
		}

		bool isTerminateSuccess = terminate();
		if (!isTerminateSuccess)
		{
			return 1;
		}

		return 0;
	}


	bool Machine::initialize()
	{
		auto entryLayer = [pLayerList = &mLayerList](std::unique_ptr<layer::BaseLayer>&& pLayer)
		{
			pLayerList->push_back(std::forward<std::unique_ptr<layer::BaseLayer>>(pLayer));
		};
#if _DEBUG
		printDoubleLine();
		std::cout << "Machine initialize start\n" << std::endl;
#endif

		entryLayer(CREATELAYER(layer::Affine, 10));
		entryLayer(CREATELAYER(layer::Affine, 20));
		entryLayer(CREATELAYER(layer::Affine, 30));

		initializeLayer();
		setupLayer();

#if _DEBUG
		std::cout << "Machine initialize finish" << std::endl;
		printDoubleLine();
#endif
		return true;

	}

	void Machine::initializeLayer()
	{
		layer::BaseLayer::flowDataFormat flowData;
		{
			flowData.batchSize	= mFlowData.batchSize;
			flowData.channel	= mFlowData.channel;
			flowData.height		= mFlowData.height;
			flowData.width		= mFlowData.width;
		}
		for (auto& layer : mLayerList)
		{
			layer->initialize(&flowData);
		}
	}
	void Machine::setupLayer()
	{
		for (auto& layer : mLayerList)
		{
			layer->setup();
		}
	}


	bool Machine::preProcess()
	{
#if _DEBUG
		std::cout << "Machine preProcess start" << std::endl;
#endif

		for (auto& pLayer : mLayerList)
		{
			pLayer->memcpyHostToDevice();
		}



#if _DEBUG
		std::cout << "Machine preProcess finish" << std::endl;
#endif
		return true;

	}

	bool Machine::mainProcess()
	{
#if _DEBUG
		std::cout << "Machine mainProcess start" << std::endl;
#endif





#if _DEBUG
		std::cout << "Machine mainProcess finish" << std::endl;
#endif
		return true;
	}

	bool Machine::postProcess()
	{
#if _DEBUG
		std::cout << "Machine postProcess start" << std::endl;
#endif

		for (auto& pLayer : mLayerList)
		{
			pLayer->memcpyDeviceToHost();
		}



#if _DEBUG
		std::cout << "Machine postProcess finish" << std::endl;
#endif
		return true;

	}

	bool Machine::terminate()
	{
#if _DEBUG
		std::cout << "Machine terminate start" << std::endl;
#endif





#if _DEBUG
		std::cout << "Machine terminate finish" << std::endl;
#endif
		return true;

	}

}