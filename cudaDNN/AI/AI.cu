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
		mOptimizer = std::move(optimizer);

		//�����֐��̓o�^
		mLossFunction = std::move(lossFunction);
		
		//�w�̃��������\�������ŕK�v�ɂȂ�p�����[�^�̐ݒ���s���B
		setupLayerInfo(shape);

		//�e�w���ɂ����郁�����̊m��
		allocLayerMemory();

	}

	constDataMemory AI::forward(f32* inputDataAddress, void* labelDataAddress)
	{
		//���̓f�[�^�̃Z�b�g
		mInputData.address = reinterpret_cast<flowDataType*>(inputDataAddress);

		//���`��
		for (auto& layer : mLayerList)
		{
			layer->forward();
		}

		//�����̌v�Z
		mLoss = mLossFunction->calcLossAndDInput(*mForwardResult, labelDataAddress);

		//�ŏI�w�̏o�͂�Ԃ��B
		return *mForwardResult;
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

		mInputData.size = shape.batchSize * shape.channel * shape.height * shape.width * sizeof(flowDataType);

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
		constDataMemory* pInputData = &mInputData;

		for (auto& layer : mLayerList)
		{
			pInputData = layer->setInputData(pInputData);
		}
		mForwardResult = pInputData;


		constDataMemory* pDInputData = &(mLossFunction->mDInputData);
		for (auto rit = mLayerList.rbegin(); rit != mLayerList.rend(); rit++)
		{
			pInputData = (*rit)->setDInputData(pInputData);
		}
	}
	



}