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

	void AI::build(InputDataInterpretation& interpretation, std::unique_ptr<optimizer::BaseOptimizer>&& optimizer, std::unique_ptr<lossFunction::BaseLossFunction>&& lossFunction)
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
		mInterpretation = interpretation;
		setupLayerInfo(mInterpretation.shape);

		//�e�w���ɂ����郁�����̊m��
		allocLayerMemory();

	}


	void AI::deepLearning(f32* pTrainingData, f32* pTrainingLabel)
	{
		mInputTrainingDataStartAddressOnCpu = pTrainingData;
		mInputTrainingLableStartAddressOnCpu = pTrainingLabel;

		mInputTrainingData.size = mInterpretation.shape.batchSize * mInterpretation.elementNum;
		mInputTrainingData.byteSize = mInterpretation.shape.batchSize * mInterpretation.byteSize;
		mInputLabelData.size = mInterpretation.shape.batchSize * 1;
		mInputLabelData.byteSize = mInterpretation.shape.batchSize * sizeof(f32);

#if _DEBUG
		mInputTrainingDataForGpuDebug.size = mInterpretation.shape.batchSize * mInterpretation.elementNum;
		mInputTrainingDataForGpuDebug.byteSize = mInterpretation.shape.batchSize * mInterpretation.byteSize;
		mInputLabelDataForGpuDebug.size = mInterpretation.shape.batchSize * 1;
		mInputLabelDataForGpuDebug.byteSize = mInterpretation.shape.batchSize * sizeof(f32);
#endif

		//
		//���`���p�̃f�[�^�������ŏ�������B
		//
		auto printer = [](std::string name, u32 value, u32 stringLen = 15)
		{
			u32 res = stringLen - name.length();
			std::cout << name + std::string(' ', res) << " = " << value << "\n";
		};
		std::cout << "TrainingData setup now" << std::endl;
		printer("TotalData num", mInterpretation.totalDataNum);
		printer("channel", mInterpretation.shape.channel);
		printer("height", mInterpretation.shape.height);
		printer("width", mInterpretation.shape.width);

		u32 loopTime = mInterpretation.totalDataNum / mInterpretation.shape.batchSize;
		for (u32 loop = 0; loop < loopTime; loop++)
		{
			u32 offset = (mInterpretation.shape.batchSize * mInterpretation.elementNum) * loop;
			if (mIsGpuAvailable)
			{
				mInputTrainingData.address = mInputTrainingDataStartAddressOnGpu + offset;
				mInputLabelData.address = mInputTrainingLableStartAddressOnGpu + offset;
#if _DEBUG
				mInputTrainingDataForGpuDebug.address = mInputTrainingDataStartAddressOnCpu + offset;
				mInputLabelDataForGpuDebug.address = mInputTrainingLableStartAddressOnCpu + offset;
#endif
			}
			else
			{
				mInputTrainingData.address = mInputTrainingDataStartAddressOnCpu + offset;
				mInputLabelData.address = mInputTrainingLableStartAddressOnCpu + offset;
			}
			forward();
#if _DEBUG
			//
			//�����Ő������`�F�b�N
			//
#endif



			backward();
#if _DEBUG
			//
			//�����Ő������`�F�b�N
			//
#endif


			optimize();
#if _DEBUG
			//
			//�����Ő������`�F�b�N
			//
#endif
		}
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

		mLossFunction->setIsGpuAvailable(mIsGpuAvailable);

		mOptimizer->setIsGpuAvailable(mIsGpuAvailable);
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
			if (mIsGpuAvailable)
			{
				layer->setInputDataForGpuDebug(pInputDataForGpuDebug);
			}
#endif
		}

		//�����֐��ɓn������A���̑��p�r�̂��߂ɏ��`���̏o�͂������ɃZ�b�g����B
		mForwardResult = pInputData;
#if _DEBUG
		if (mIsGpuAvailable)
		{
			mForwardResultForGpuDebug = pInputDataForGpuDebug;
		}
#endif

		//�����֐��ɏ��`���̌��ʂ�n���B
		mLossFunction->setInput(mForwardResult, &mInputLabelData);
#if _DEBUG
		if (mIsGpuAvailable)
		{
			mLossFunction->setInputForGpuDebug(mForwardResultForGpuDebug, &mInputLabelDataForGpuDebug);
		}
#endif


		DataMemory* pDInputData = &(mLossFunction->mDInputData);
#if _DEBUG
		DataMemory* pDInputDataForGpuDebug = nullptr;
		if (mIsGpuAvailable)
		{
			pDInputDataForGpuDebug = &(mLossFunction->mDInputDataForGpuDebug);
		}
#endif
		for (auto rit = mLayerList.rbegin(); rit != mLayerList.rend(); rit++)
		{
			(*rit)->setDInputData(pDInputData);
#if _DEBUG
			if (mIsGpuAvailable)
			{
				(*rit)->setDInputDataForGpuDebug(pDInputDataForGpuDebug);
			}
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
		mLoss = mLossFunction->calcLossAndDInput();
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