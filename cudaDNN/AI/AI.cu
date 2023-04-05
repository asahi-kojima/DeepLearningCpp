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


	void AI::deepLearning(f32* pTrainingData, f32* pTrainingLabel, u32 epochs, f32 learningRate)
	{
		mOptimizer->setLearningRate(learningRate);

		dataSetup(pTrainingData, pTrainingLabel);

		//
		//���`���p�̃f�[�^�������ŏ�������B
		//
		auto printer = [](std::string name, u32 value, u32 stringLen = 15)
		{
			u32 res = stringLen - name.length();
			std::string space = std::string(res, ' ');
			std::cout << name << space << " = " << value << "\n";
		};
		std::cout << "TrainingData setup now" << std::endl;
		printer("TotalData num", mInterpretation.totalDataNum);
		printer("channel", mInterpretation.shape.channel);
		printer("height", mInterpretation.shape.height);
		printer("width", mInterpretation.shape.width);

		u32 loopTime = mInterpretation.totalDataNum / mInterpretation.shape.batchSize;

		for (u32 epoch = 0; epoch < epochs; epoch++)
		{
			std::cout << "epoch = " << epoch + 1 << std::endl;
			f32 loss = 0.0f;
			for (u32 loop = 0; loop < loopTime; loop++)
			{
				u32 offsetForData = (mInterpretation.shape.batchSize * mInterpretation.elementNum) * loop;
				u32 offsetForLabel = (mInterpretation.shape.batchSize * 1) * loop;
				if (mIsGpuAvailable)
				{
					mInputTrainingDataOnGPU.address = mInputTrainingDataStartAddressOnGPU + offsetForData;
					mInputLabelDataOnGPU.address = mInputTrainingLableStartAddressOnGPU + offsetForLabel;
				}
				else
				{
					mInputTrainingDataOnCPU.address = mInputTrainingDataStartAddressOnCPU + offsetForData;
					mInputLabelDataOnCPU.address = mInputTrainingLableStartAddressOnCPU + offsetForLabel;
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
				if (mIsGpuAvailable)
					loss += mLossOnGPU;
				else
					loss += mLossOnCPU;
			}
			std::cout << "current loss = " << loss / loopTime << std::endl;
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
		std::cout << std::string(100, '=') << std::endl;
		std::cout << "GPU Resource Check..." << std::endl;
		std::cout << std::string(100, '=') << std::endl;
		int deviceCount = 0;
		cudaGetDeviceCount(&deviceCount);

		if (deviceCount == 0)
		{
			mIsGpuAvailable = false;
			printf("There are no available device(s) that support CUDA\n");
		}
		else
		{
			mIsGpuAvailable = true;
			printf("Detected %d CUDA Capable device(s)\n", deviceCount);
		}

		int dev = 0, driverVersion = 0, runtimeVersion = 0;
		CHECK(cudaSetDevice(dev));
		cudaDeviceProp deviceProp;
		CHECK(cudaGetDeviceProperties(&deviceProp, dev));
		printf("Device %d: \"%s\"\n", dev, deviceProp.name);

		cudaDriverGetVersion(&driverVersion);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
			driverVersion / 1000, (driverVersion % 100) / 10,
			runtimeVersion / 1000, (runtimeVersion % 100) / 10);
		printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
			deviceProp.major, deviceProp.minor);
		printf("  Total amount of global memory:                 %.2f GBytes (%llu "
			"bytes)\n", (float)deviceProp.totalGlobalMem / pow(1024.0, 3),
			(unsigned long long)deviceProp.totalGlobalMem);
		printf("  GPU Clock rate:                                %.0f MHz (%0.2f "
			"GHz)\n", deviceProp.clockRate * 1e-3f,
			deviceProp.clockRate * 1e-6f);
		printf("  Memory Clock rate:                             %.0f Mhz\n",
			deviceProp.memoryClockRate * 1e-3f);
		printf("  Memory Bus Width:                              %d-bit\n",
			deviceProp.memoryBusWidth);

		if (deviceProp.l2CacheSize)
		{
			printf("  L2 Cache Size:                                 %d bytes\n",
				deviceProp.l2CacheSize);
		}

		printf("  Max Texture Dimension Size (x,y,z)             1D=(%d), "
			"2D=(%d,%d), 3D=(%d,%d,%d)\n", deviceProp.maxTexture1D,
			deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
			deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1],
			deviceProp.maxTexture3D[2]);
		printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, "
			"2D=(%d,%d) x %d\n", deviceProp.maxTexture1DLayered[0],
			deviceProp.maxTexture1DLayered[1], deviceProp.maxTexture2DLayered[0],
			deviceProp.maxTexture2DLayered[1],
			deviceProp.maxTexture2DLayered[2]);
		printf("  Total amount of constant memory:               %lu bytes\n",
			deviceProp.totalConstMem);
		printf("  Total amount of shared memory per block:       %lu bytes\n",
			deviceProp.sharedMemPerBlock);
		printf("  Total number of registers available per block: %d\n",
			deviceProp.regsPerBlock);
		printf("  Warp size:                                     %d\n",
			deviceProp.warpSize);
		printf("  Maximum number of threads per multiprocessor:  %d\n",
			deviceProp.maxThreadsPerMultiProcessor);
		printf("  Maximum number of threads per block:           %d\n",
			deviceProp.maxThreadsPerBlock);
		printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
			deviceProp.maxThreadsDim[0],
			deviceProp.maxThreadsDim[1],
			deviceProp.maxThreadsDim[2]);
		printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
			deviceProp.maxGridSize[0],
			deviceProp.maxGridSize[1],
			deviceProp.maxGridSize[2]);
		printf("  Maximum memory pitch:                          %lu bytes\n",
			deviceProp.memPitch);
		std::cout << std::string(100, '=') << "\n\n" << std::endl;
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


		for (auto& layer : mLayerList)
		{
			layer->setupLayerInfo(&dataShape);
		}

		mLossFunction->setupDataShape(dataShape);

		if (mIsGpuAvailable)
		{
			mInputTrainingDataOnGPU.size = shape.batchSize * shape.channel * shape.height * shape.width;
		}
		else
		{
			mInputTrainingDataOnCPU.size = shape.batchSize * shape.channel * shape.height * shape.width;
		}
	}

	/// <summary>
	///�@�e�w�ɂ�����p�����[�^�̂��߂̃������m�ۂ⏉�����A
	/// �����Ċw�K���Ɋe�w���K�v�ƂȂ�O�̑w�̏o�̓f�[�^�̃A�h���X��o�^�B
	/// </summary>
	void AI::allocLayerMemory()
	{
		if (mIsGpuAvailable)
		{
			//GPU��̃������̊m�ۂ₻��̏�����
			for (auto& layer : mLayerList)
			{
				layer->initializeOnGPU();
			}

			mLossFunction->initializeOnGPU();

			//�w�K���̊e�w���Q�Ƃ���O�w�̃f�[�^�̃A�h���X��o�^
			//�܂���_�ƂȂ�f�[�^���Z�b�g
			DataMemory* pInputDataOnGPU = &mInputTrainingDataOnGPU;

			//�����Ŋe�w�ɎQ�Ƃ���ׂ��f�[�^�����ɓn���Ă����B
			for (auto& layer : mLayerList)
			{
				layer->setInputDataOnGPU(pInputDataOnGPU);
			}

			//�����֐��ɓn������A���̑��p�r�̂��߂ɏ��`���̏o�͂������ɃZ�b�g����B
			mForwardResultOnGPU = pInputDataOnGPU;

			//�����֐��ɏ��`���̌��ʂ�n���B
			mLossFunction->setInputOnGPU(mForwardResultOnGPU, &mInputLabelDataOnGPU);


			DataMemory* pDInputDataOnGPU = mLossFunction->getDInputDataOnGPU();

			for (auto rit = mLayerList.rbegin(); rit != mLayerList.rend(); rit++)
			{
				(*rit)->setDInputDataOnGPU(pDInputDataOnGPU);
			}
		}
		else
		{
			//GPU��̃������̊m�ۂ₻��̏�����
			for (auto& layer : mLayerList)
			{
				layer->initializeOnCPU();
			}

			mLossFunction->initializeOnCPU();

			//�w�K���̊e�w���Q�Ƃ���O�w�̃f�[�^�̃A�h���X��o�^
			//�܂���_�ƂȂ�f�[�^���Z�b�g
			DataMemory* pInputDataOnCPU = &mInputTrainingDataOnCPU;

			//�����Ŋe�w�ɎQ�Ƃ���ׂ��f�[�^�����ɓn���Ă����B
			for (auto& layer : mLayerList)
			{
				layer->setInputDataOnCPU(pInputDataOnCPU);
			}

			//�����֐��ɓn������A���̑��p�r�̂��߂ɏ��`���̏o�͂������ɃZ�b�g����B
			mForwardResultOnCPU = pInputDataOnCPU;

			//�����֐��ɏ��`���̌��ʂ�n���B
			mLossFunction->setInputOnCPU(mForwardResultOnCPU, &mInputLabelDataOnCPU);


			DataMemory* pDInputDataOnCPU = mLossFunction->getDInputDataOnCPU();

			for (auto rit = mLayerList.rbegin(); rit != mLayerList.rend(); rit++)
			{
				(*rit)->setDInputDataOnCPU(pDInputDataOnCPU);
			}
		}
	}

	void AI::dataSetup(f32* pTrainingData, f32* pTrainingLabel)
	{
		mInputTrainingDataStartAddressOnCPU = pTrainingData;
		mInputTrainingLableStartAddressOnCPU = pTrainingLabel;

		if (mIsGpuAvailable)
		{
			mInputTrainingDataOnGPU.size = mInterpretation.shape.batchSize * mInterpretation.elementNum;
			mInputTrainingDataOnGPU.byteSize = mInterpretation.shape.batchSize * mInterpretation.byteSize;
			mInputLabelDataOnGPU.size = mInterpretation.shape.batchSize * 1;
			mInputLabelDataOnGPU.byteSize = mInterpretation.shape.batchSize * sizeof(f32);

			CHECK(cudaMalloc((void**)(&mInputTrainingDataStartAddressOnGPU), 60000 * 1 * 28 * 28 * sizeof(f32)));
			CHECK(cudaMalloc((void**)(&mInputTrainingLableStartAddressOnGPU), 60000 * 1 * 1 * 1 * sizeof(f32)));

			CHECK(cudaMemcpy(mInputTrainingDataStartAddressOnGPU, mInputTrainingDataStartAddressOnCPU, 60000 * 1 * 28 * 28 * sizeof(f32), cudaMemcpyHostToDevice));
			CHECK(cudaMemcpy(mInputTrainingLableStartAddressOnGPU, mInputTrainingLableStartAddressOnCPU, 60000 * 1 * 1 * 1 * sizeof(f32), cudaMemcpyHostToDevice));
		}
		else
		{
			mInputTrainingDataOnCPU.size = mInterpretation.shape.batchSize * mInterpretation.elementNum;
			mInputTrainingDataOnCPU.byteSize = mInterpretation.shape.batchSize * mInterpretation.byteSize;
			mInputLabelDataOnCPU.size = mInterpretation.shape.batchSize * 1;
			mInputLabelDataOnCPU.byteSize = mInterpretation.shape.batchSize * sizeof(f32);
		}
	}

	void AI::forward()
	{
		if (mIsGpuAvailable)
		{
			//���`��
			for (auto& layer : mLayerList)
			{
				layer->forwardOnGPU();
			}

			//�����̌v�Z
			mLossOnGPU = mLossFunction->calcLossAndDInputOnGPU();
		}
		else
		{
			//���`��
			for (auto& layer : mLayerList)
			{
				layer->forwardOnCPU();
			}

			//�����̌v�Z
			mLossOnCPU = mLossFunction->calcLossAndDInputOnCPU();
		}
	}

	void AI::backward()
	{
		if (mIsGpuAvailable)
		{
			for (auto riter = mLayerList.rbegin(), end = mLayerList.rend(); riter != end; riter++)
			{
				(*riter)->backwardOnGPU();
			}
		}
		else
		{
			for (auto riter = mLayerList.rbegin(), end = mLayerList.rend(); riter != end; riter++)
			{
				(*riter)->backwardOnCPU();
			}
		}
	}


	void AI::optimize()
	{
		if (mIsGpuAvailable)
		{
			for (auto& layer : mLayerList)
			{
				mOptimizer->optimizeOnGPU(layer);
			}
		}
		else
		{
			for (auto& layer : mLayerList)
			{
				mOptimizer->optimizeOnCPU(layer);
			}
		}

	}

#pragma endregion
}