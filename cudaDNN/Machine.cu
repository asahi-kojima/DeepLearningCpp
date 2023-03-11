#include <iostream>
#include <thread>

#include "./Layer/Layer.h"
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
		printDoubleLine();
		std::cout << "Machine initialize start\n" << std::endl;


		entryLayer(CREATELAYER(layer::Affine, 10));
		entryLayer(CREATELAYER(layer::Affine, 20));
		entryLayer(CREATELAYER(layer::Affine, 30));
		entryLayer(CREATELAYER(layer::Affine, 40));
		entryLayer(CREATELAYER(layer::Affine, 50));

		initializeLayer();
		setupLayer();


		std::cout << "Machine initialize finish" << std::endl;
		printDoubleLine();

		return true;

	}

	void Machine::entryLayer(std::unique_ptr<layer::BaseLayer>&& pLayer)
	{
		mLayerList.push_back(std::forward<std::unique_ptr<layer::BaseLayer>>(pLayer));
	}

	/// <summary>
	/// �e�w�̓����p�����[�^���v�Z����B
	/// flowData�ɂ͓��̓f�[�^�̌`�󂪓����Ă���̂ŁA
	/// �������ɃJ�[�l���̃T�C�Y��p�����[�^�̐����v�Z�B
	/// </summary>
	void Machine::initializeLayer()
	{
		layer::BaseLayer::flowDataFormat flowDataShape;
		{
			flowDataShape.batchSize	= mFlowData.batchSize;
			flowDataShape.channel	= mFlowData.channel;
			flowDataShape.height		= mFlowData.height;
			flowDataShape.width		= mFlowData.width;
		}
		for (auto& layer : mLayerList)
		{
			layer->initialize(&flowDataShape);
		}
	}

	/// <summary>
	///�@�e�w�ɂ�����p�����[�^�̂��߂̃������m�ۂ⏉�����A
	/// �����Ċw�K���Ɋe�w���K�v�ƂȂ�O�̑w�̏o�̓f�[�^�̃A�h���X��o�^�B
	/// </summary>
	void Machine::setupLayer()
	{
		//GPU��̃������̊m�ۂ₻��̏�����
		for (auto& layer : mLayerList)
		{
			layer->setup();
		}

		//�w�K���̊e�w���Q�Ƃ���O�w�̃f�[�^�̃A�h���X��o�^
		dataMemory* pInputData = &mLearningData;

		for (auto& layer : mLayerList)
		{
			layer->setInputDataOnGPU(pInputData);
			pInputData = layer->getDataMemory();
		}
		for (auto rit = mLayerList.rbegin(); rit != mLayerList.rend(); rit++)
		{
			(*rit)->setDInputDataOnGPU(pInputData);
			pInputData = (*rit)->getDDataMemory();
		}

		//�f�o�b�O�p

	}


	bool Machine::preProcess()
	{
#if _DEBUG
		std::cout << "Machine preProcess start" << std::endl;
#endif

		makeTestData();

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
		for (auto& layer : mLayerList)
		{
			layer->forward();
		}

		for (auto rit = mLayerList.rbegin(); rit != mLayerList.rend(); rit++)
		{
			(*rit)->backward();
		}


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


	void Machine::makeTestData()
	{
		mLearningData.dataNum = (mFlowData.batchSize * mFlowData.width);
		flowDataType* tmp = new flowDataType[mLearningData.dataNum];
		for (u32 idx = 0; idx < mLearningData.dataNum; idx++)
		{
			tmp[idx] = 0.1f;
		}

		CHECK(cudaMalloc((void**)(&(mLearningData.dataAddress)), mLearningData.dataNum * sizeof(f32)));
		CHECK(cudaMemcpy(mLearningData.dataAddress, tmp, mLearningData.dataNum * sizeof(f32), cudaMemcpyHostToDevice));

		delete[] tmp;
	}

}