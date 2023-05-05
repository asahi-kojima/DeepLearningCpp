#include <iostream>
#include <thread>
#include <random>
#include <cassert>

#include "./Layer/Layer.h"
#include "./Optimizer/Optimizer.h"
#include "AI.h"
#include "AIMacro.h"
#include "../commonGPU.cuh"
#include "../commonCPU.h"



namespace Aoba
{
#pragma region public

	AI::AI() = default;
	AI::~AI() = default;


	void AI::build(DataFormat4DeepLearning& format)
	{
		//------------------------------------------------------------------
		//�w���Œ�ł�����邩�̃`�F�b�N
		//------------------------------------------------------------------
		assert(mLayerList.size() > 0);

		//------------------------------------------------------------------
		//�I�v�e�B�}�C�U�[�̓o�^
		//------------------------------------------------------------------
		if (mOptimizer == nullptr)
		{
			std::cout << "Optimizer is not defined. Default Optimizer is set..." << std::endl;
			mOptimizer = std::make_unique<optimizer::Sgd>(0.001f);
		}

		//------------------------------------------------------------------
		//�����֐��̓o�^
		//------------------------------------------------------------------
		if (mLossFunction == nullptr)
		{
			std::cout << "Loss function is not defined. Default Loss function is set..." << std::endl;
			mLossFunction = std::make_unique<lossFunction::CrossEntropyWithSM>();
		}

		//------------------------------------------------------------------
		//GPU�̗��p���\���`�F�b�N���A�����o�́B�܂��S�Ă̑w�ɂ��̏��𑗂�B
		//------------------------------------------------------------------
		checkGpuIsAvailable();

		//------------------------------------------------------------------
		//�f�[�^�t�H�[�}�b�g��ۑ�
		//------------------------------------------------------------------
		mDataFormat4DeepLearning = format;

		//------------------------------------------------------------------
		// �e�w�ɂ�������̓f�[�^�̌`��o�^
		// �y�сA�w�̃��������\�������ŕK�v�ɂȂ�p�����[�^�̐ݒ���s���B
		//------------------------------------------------------------------
		initializeLayer();
	}


	void AI::deepLearning(f32* pTrainingData, f32* pCorrectData, u32 epochs)
	{
		//mOptimizer->setLearningRate(learningRate);


		//------------------------------------------------------------------
		//�w�̏���\��
		//------------------------------------------------------------------
		std::cout << "[ Layer Information ]" << std::endl;
		for (auto& layer : mLayerList)
		{
			layer->printLayerInfo();
		}
		std::cout << std::endl;
		
		
		//------------------------------------------------------------------
		//�P���f�[�^�̏���\��
		//------------------------------------------------------------------
		std::cout << "[ Training Data Information ]" << std::endl;
#pragma region print_TrainingData_infomation
		auto printer = [](std::string name, u32 value, u32 stringLen = 15)
		{
			u32 res = stringLen - name.length();
			std::string space = std::string(res, ' ');
			std::cout << name << space << " = " << value << "\n";
		};
		printer("TotalData num", mDataFormat4DeepLearning.dataNum);
		printer("channel", mDataFormat4DeepLearning.trainingDataShape.channel);
		printer("height", mDataFormat4DeepLearning.trainingDataShape.height);
		printer("width", mDataFormat4DeepLearning.trainingDataShape.width);
		std::cout << std::endl;
#pragma endregion


		//------------------------------------------------------------------
		//���`���p�̃f�[�^�������ŏ�������B
		//------------------------------------------------------------------
		dataSetup(pTrainingData, pCorrectData);


		//------------------------------------------------------------------
		//�[�w�w�K�̎��s�ӏ�
		//------------------------------------------------------------------
		std::cout << "[ Deep Learning Start ]" << std::endl;
		u32 loopTime = mDataFormat4DeepLearning.dataNum / mDataFormat4DeepLearning.batchSize;
		u32 batch = mDataFormat4DeepLearning.batchSize;
		auto progressBar = [](u32 currentLoop, u32 totalLoop, u32 length = 20)
		{
			u32 grid = totalLoop / length;
			std::string s = "\r";
			for (u32 i = 0; i < static_cast<u32>((static_cast<f32>(length) * currentLoop) / totalLoop); i++)
			{
				s += "=";
			}
			s += ">";
			s32 spaceLength = static_cast<s32>(length - s.length() + 2);
			for (s32 i = 0; i < spaceLength; i++)
			{
				s += " ";
			}
			s += " " + std::to_string(static_cast<u32>(static_cast<f32>(currentLoop * 100) / totalLoop)) + "/100";
			printf(s.c_str());
		};

		for (u32 epoch = 0; epoch < epochs; epoch++)
		{
			std::cout << "epoch = " << epoch + 1 << std::endl;
			f32 loss = 0.0f;
			std::cout << "deep learning now" << std::endl;
			for (u32 loop = 0; loop < loopTime; loop++)
			{
				progressBar(loop + 1, loopTime);
				u32 offsetForTrainingData = (batch * mDataFormat4DeepLearning.eachTrainingDataSize) * loop;
				u32 offsetForCorrectData = (batch * mDataFormat4DeepLearning.eachCorrectDataSize) * loop;
				if (mIsGpuAvailable)
				{
					mInputTrainingDataOnGPU.address = mInputTrainingDataStartAddressOnGPU + offsetForTrainingData;
					mInputCorrectDataOnGPU.address = mInputCorrectDataStartAddressOnGPU + offsetForCorrectData;
				}
				else
				{
					mInputTrainingDataOnCPU.address = mInputTrainingDataStartAddressOnCPU + offsetForTrainingData;
					mInputCorrectDataOnCPU.address = mInputCorrectDataStartAddressOnCPU + offsetForCorrectData;
				}


				forward();
#if _DEBUG
				//------------------------------------------------------------------
				//�����Ő������`�F�b�N
				//------------------------------------------------------------------
#endif



				backward();
#if _DEBUG
				//------------------------------------------------------------------
				//�����Ő������`�F�b�N
				//------------------------------------------------------------------
#endif


				optimize();
#if _DEBUG
				//------------------------------------------------------------------
				//�����Ő������`�F�b�N
				//------------------------------------------------------------------
#endif
				if (mIsGpuAvailable)
					loss += mLossOnGPU;
				else
					loss += mLossOnCPU;
			}
			std::cout << "\n";
			std::cout << "current loss = " << loss / loopTime << "\n" << std::endl;
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
		std::cout << "[ GPU Information ]" << std::endl;
		mIsGpuAvailable = true;
		std::cout <<  std::endl;
	}


	/// <summary>
	///�@�e�w�ɂ�����p�����[�^�̂��߂̃������m�ۂ⏉�����A
	/// �����Ċw�K���Ɋe�w���K�v�ƂȂ�O�̑w�̏o�̓f�[�^�̃A�h���X��o�^�B
	/// </summary>
	void AI::initializeLayer()
	{
		if (mIsGpuAvailable)
		{
			INITIALIZE_ON_(GPU);
		}
		else
		{
			INITIALIZE_ON_(CPU);
		}
	}

	void AI::dataSetup(f32* pTrainingData, f32* pCorrectData)
	{
		mInputTrainingDataStartAddressOnCPU = pTrainingData;
		mInputCorrectDataStartAddressOnCPU = pCorrectData;

		u32 batch = mDataFormat4DeepLearning.batchSize;
		if (mIsGpuAvailable)
		{
			mInputTrainingDataOnGPU.size = batch * mDataFormat4DeepLearning.eachTrainingDataSize;
			mInputTrainingDataOnGPU.byteSize = mInputTrainingDataOnGPU.size * sizeof(f32);
			mInputCorrectDataOnGPU.size = batch * mDataFormat4DeepLearning.eachCorrectDataSize;
			mInputCorrectDataOnGPU.byteSize = mInputCorrectDataOnGPU.size * sizeof(f32);

			CHECK(cudaMalloc((void**)(&mInputTrainingDataStartAddressOnGPU), mDataFormat4DeepLearning.dataNum * mDataFormat4DeepLearning.eachTrainingDataSize * sizeof(f32)));
			CHECK(cudaMalloc((void**)(&mInputCorrectDataStartAddressOnGPU), mDataFormat4DeepLearning.dataNum * mDataFormat4DeepLearning.eachCorrectDataSize * sizeof(f32)));

			CHECK(cudaMemcpy(mInputTrainingDataStartAddressOnGPU, mInputTrainingDataStartAddressOnCPU, mDataFormat4DeepLearning.dataNum * mDataFormat4DeepLearning.eachTrainingDataSize * sizeof(f32), cudaMemcpyHostToDevice));
			CHECK(cudaMemcpy(mInputCorrectDataStartAddressOnGPU, mInputCorrectDataStartAddressOnCPU, mDataFormat4DeepLearning.dataNum * mDataFormat4DeepLearning.eachCorrectDataSize * sizeof(f32), cudaMemcpyHostToDevice));
		}
		else
		{
			mInputTrainingDataOnCPU.size = batch * mDataFormat4DeepLearning.eachTrainingDataSize;
			mInputTrainingDataOnCPU.byteSize = mInputTrainingDataOnCPU.size * sizeof(f32);
			mInputCorrectDataOnCPU.size = batch * mDataFormat4DeepLearning.eachCorrectDataSize;
			mInputCorrectDataOnCPU.byteSize = mInputCorrectDataOnCPU.size * sizeof(f32);
		}
	}

	void AI::forward()
	{
		if (mIsGpuAvailable)
		{
			FORWARD_ON_(GPU);
		}
		else
		{
			FORWARD_ON_(CPU);
		}
	}

	void AI::backward()
	{
		if (mIsGpuAvailable)
		{
			BACKWARD_ON_(GPU);
		}
		else
		{
			BACKWARD_ON_(CPU);
		}
	}


	void AI::optimize()
	{
		if (mIsGpuAvailable)
		{
			OPTIMIZE_ON_(GPU);
		}
		else
		{
			OPTIMIZE_ON_(CPU);
		}
	}

#pragma endregion
}