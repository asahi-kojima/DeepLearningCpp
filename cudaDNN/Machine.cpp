#include <iostream>
#include <thread>
#include "Layer.h"
#include "Machine.h"


#define CREATELAYER(classname, ...) std::make_unique<classname>(__VA_ARGS__)


namespace miduho
{
	u32 Machine::entry()
	{
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
		std::cout << "Machine initialize start" << std::endl;
#endif


		entryLayer(CREATELAYER(layer::TestLayer));
		entryLayer(CREATELAYER(layer::TestLayer, 1));



#if _DEBUG
		std::cout << "Machine initialize finish" << std::endl;
#endif
		return true;

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