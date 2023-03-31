#include <iostream>
#include <thread>
#include <random>
#include <cassert>

#include "./Layer/Layer.h"
#include "./Optimizer/Optimizer.h"
#include "AI.h"
#include "../commonGPU.cuh"
#include "../commonCPU.h"



namespace Aoba
{
#pragma region public

	AI::AI() = default;
	AI::~AI() = default;

	void AI::addLayer(std::unique_ptr<layer::BaseLayer>&& pLayer)
	{
		mLayerList.push_back(std::forward<std::unique_ptr<layer::BaseLayer>>(pLayer));
	}

	void AI::build(InputDataShape& shape, std::unique_ptr<optimizer::BaseOptimizer>&& optimizer, std::unique_ptr<lossFunction::BaseLossFunction>&& lossFunction)
	{
		//�w���Œ�ł�����邩�̃`�F�b�N
		assert(mLayerList.size() > 0);

		//�I�v�e�B�}�C�U�[�̓o�^
		assert(optimizer != nullptr);
		mOptimizer = std::move(optimizer);

		//�����֐��̓o�^
		assert(lossFunction != nullptr);
		mLossFunction = std::move(lossFunction);

		//GPU�̗��p���\���`�F�b�N���A�����o�́B�܂��S�Ă̑w�ɂ��̏��𑗂�B
		checkGpuIsAvailable();

		//�w�̃��������\�������ŕK�v�ɂȂ�p�����[�^�̐ݒ���s���B
		setupLayerInfo(shape);

		//�e�w���ɂ����郁�����̊m��
		allocLayerMemory();

	}

	void AI::setLearningData(DataMemory*, DataMemory*)
	{
	}

	void AI::deepLearning()
	{
		//
		//���`���p�̃f�[�^�������ŏ�������B
		//


		forward();
		backward();
		optimize();
	}


	DataMemory AI::operator()(f32* inputData)
	{
		return DataMemory();
	}

#pragma endregion

#pragma region private
	void AI::checkGpuIsAvailable()
	{
		//GPU���p�\���̏���������B
		mIsGpuAvailable = true;


		for (auto& layer : mLayerList)
		{
			layer->setIsGpuAvailable(mIsGpuAvailable);
		}
	}

	/// <summary>
	/// �e�w�̓����p�����[�^���v�Z����B
	/// flowDataShape�ɂ͓��̓f�[�^�̌`�󂪓����Ă���̂ŁA
	/// �������ɃJ�[�l���̃T�C�Y��p�����[�^�̐����v�Z�B
	/// </summary>
	void AI::setupLayerInfo(InputDataShape& shape)
	{
		InputDataShape dataShape;
		{
			dataShape.batchSize = shape.batchSize;
			dataShape.channel = shape.channel;
			dataShape.height = shape.height;
			dataShape.width = shape.width;
		}

		mInputTrainingData.size = shape.batchSize * shape.channel * shape.height * shape.width;
#if _DEBUG
		if (mIsGpuAvailable)
		{
			mInputTrainingDataForGpuDebug.size = shape.batchSize * shape.channel * shape.height * shape.width;
		}
#endif

		for (auto& layer : mLayerList)
		{
			layer->setupLayerInfo(&dataShape);
		}

		mLossFunction->setupDataShape(dataShape);
	}

	/// <summary>
	///�@�e�w�ɂ�����p�����[�^�̂��߂̃������m�ۂ⏉�����A
	/// �����Ċw�K���Ɋe�w���K�v�ƂȂ�O�̑w�̏o�̓f�[�^�̃A�h���X��o�^�B
	/// </summary>
	void AI::allocLayerMemory()
	{
		//GPU��̃������̊m�ۂ₻��̏�����
		for (auto& layer : mLayerList)
		{
			layer->initialize();
		}

		mLossFunction->initialize();

		//�w�K���̊e�w���Q�Ƃ���O�w�̃f�[�^�̃A�h���X��o�^
		//�܂���_�ƂȂ�f�[�^���Z�b�g
		DataMemory* pInputData = &mInputTrainingData;
#if _DEBUG
		DataMemory* pInputDataForGpuDebug = &mInputTrainingDataForGpuDebug;
#endif

		//�����Ŋe�w�ɎQ�Ƃ���ׂ��f�[�^�����ɓn���Ă����B
		for (auto& layer : mLayerList)
		{
			layer->setInputData(pInputData);
#if _DEBUG
			layer->setInputDataForGpuDebug(pInputDataForGpuDebug);
#endif
		}

		//�����֐��ɓn������A���̑��p�r�̂��߂ɏ��`���̏o�͂������ɃZ�b�g����B
		mForwardResult = pInputData;
#if _DEBUG
		mForwardResultForGpuDebug = pInputDataForGpuDebug;
#endif


		DataMemory* pDInputData = &(mLossFunction->mDInputData);
#if _DEBUG
		DataMemory* pDInputDataForGpuDebug = &(mLossFunction->mDInputDataForGpuDebug);
#endif
		for (auto rit = mLayerList.rbegin(); rit != mLayerList.rend(); rit++)
		{
			(*rit)->setDInputData(pDInputData);
#if _DEBUG
			(*rit)->setDInputDataForGpuDebug(pDInputDataForGpuDebug);
#endif
		}
	}

	void AI::forward()
	{
		//���`��
		for (auto& layer : mLayerList)
		{
			layer->forward();
		}

		//�����̌v�Z
		mLoss = mLossFunction->calcLossAndDInput(*mForwardResult, mInputLabelData.address);
	}

	void AI::backward()
	{
		for (auto riter = mLayerList.rbegin(), end = mLayerList.rend(); riter != end; riter++)
		{
			(*riter)->backward();
		}
	}


	void AI::optimize()
	{
		for (auto& layer : mLayerList)
		{
			mOptimizer->optimize(layer);
		}
	}

#pragma endregion
}