#include <random>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

#include "Affine.h"
#include "../../../commonCPU.h"

namespace Aoba::layer
{


	Affine::Affine(u32 outputSize, f32 weight)
		:mBatchSize(0)
		,mInputSize(0)
		,mOutputSize(outputSize)
		,mAffineParamWeight(weight)
	{
	}

	Affine::~Affine()
	{
	}

	void Affine::setupLayerInfo(DataShape* pInputData)
	{
		mBatchSize = pInputData->batchSize;
		mInputSize = pInputData->width;

		pInputData->width = mOutputSize;

		mAlreadySetup = true;
	}


	
	void Affine::memcpyHostToDevice()
	{

	}

	void Affine::memcpyDeviceToHost()
	{

	}




	//////////////////////////////////////
	//CPU �֐�
	//////////////////////////////////////
	void Affine::initializeOnCPU()
	{
		pParametersOnCPU.resize(2);

		//Affine/dAffine�p�����[�^
		//(1)�Q��
		paramMemory& affineParam = pParametersOnCPU[0];
		paramMemory& affineDParam = pDParametersOnCPU[0];
		//(2)�p�����[�^�̃T�C�Y��ݒ�
		affineParam.size = mOutputSize * mInputSize;
		affineDParam.size = mOutputSize * mInputSize;
		//(3)�p�����[�^�p�̗̈�m��
		affineParam.address = new parameterType[affineParam.size];
		affineDParam.address = new parameterType[affineDParam.size];
		//(4)������
		{
			std::random_device seed_gen;
			std::default_random_engine engine(seed_gen());
			std::normal_distribution<> dist(0.0, std::sqrt(2.0 / mInputSize));
			for (u32 idx = 0; idx < affineParam.size; idx++)
			{
				affineParam.address[idx] = mAffineParamWeight * static_cast<f32>(dist(engine)) / std::sqrt(2.0f / mInputSize);
			}

			for (u32 idx = 0; idx < affineDParam.size; idx++)
			{
				affineDParam.address[idx] = 0.0f;
			}
		}
		//Bias�p�����[�^
		//(1)�Q��
		paramMemory& biasParam = pParametersOnCPU[1];
		paramMemory& biasDParam = pDParametersOnCPU[1];
		//(2)�p�����[�^�̃T�C�Y��ݒ�
		biasParam.size = mOutputSize;
		biasDParam.size = mOutputSize;
		//(3)�p�����[�^�p�̗̈�m��
		biasParam.address = new parameterType[biasParam.size];
		biasDParam.address = new parameterType[biasDParam.size];
		//(4)������
		{
			for (u32 idx = 0; idx < biasParam.size; idx++)
			{
				biasParam.address[idx] = 0.0f;
			}

			for (u32 idx = 0; idx < biasDParam.size; idx++)
			{
				biasDParam.address[idx] = 0.0f;
			}
		}
	}

	void Affine::forwardOnCPU()
	{

	}

	void Affine::backwardOnCPU()
	{

	}

	void Affine::terminateOnCPU()
	{

	}
} 