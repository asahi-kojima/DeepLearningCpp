#include <iostream>
#include <thread>
#include <random>
#include <cassert>
#include "AIHelperFunction.h"
#include "AI.h"
#include "AIMacro.h"
#include "./Layer/Layer.h"
#include "./Optimizer/Optimizer.h"


std::map<std::string, f32> timers;

namespace Aoba
{
#pragma region public

	AI::AI() = default;
	AI::~AI()
	{
		if (!mAlreadyBuild)
		{
			return;
		}


		if (mIsGpuAvailable)
		{
			mOptimizer->terminateOnGPU();
			mLossFunction->terminateOnGPU();
			for (auto& layer : mLayerList)
			{
				layer->terminateOnGPU();
			}
		}
		else
		{
			mOptimizer->terminateOnCPU();
			mLossFunction->terminateOnCPU();
			for (auto& layer : mLayerList)
			{
				layer->terminateOnCPU();
			}
		}
	}


	void AI::build(DataFormat4DeepLearning& format)
	{
		//------------------------------------------------------------------
		//層が最低でも一つあるかのチェック
		//------------------------------------------------------------------
		assert(mLayerList.size() > 0);



		//------------------------------------------------------------------
		//オプティマイザーの登録
		//------------------------------------------------------------------
		if (mOptimizer == nullptr)
		{
			std::cout << "Optimizer is not defined. Default Optimizer is set..." << std::endl;
			mOptimizer = std::make_unique<optimizer::Sgd>(0.001f);
		}

		//------------------------------------------------------------------
		//損失関数の登録
		//------------------------------------------------------------------
		if (mLossFunction == nullptr)
		{
			std::cout << "Loss function is not defined. Default Loss function is set..." << std::endl;
			mLossFunction = std::make_unique<lossFunction::CrossEntropyWithSM>();
		}

		//------------------------------------------------------------------
		//GPUの利用が可能かチェックし、情報を出力。
		//------------------------------------------------------------------
		checkGpuIsAvailable();

		//------------------------------------------------------------------
		//データフォーマットを保存
		//------------------------------------------------------------------
		mDataFormat4DeepLearning = format;
		assert(format.batchSize != 0);

		//------------------------------------------------------------------
		// 各層における入力データの形状登録
		// 及び、層のメモリを構成する上で必要になるパラメータの設定を行う。
		//------------------------------------------------------------------
		initializeLayer();

		//------------------------------------------------------------------
		//optimizerの初期化
		//------------------------------------------------------------------
		initializeOptimizer();

		//------------------------------------------------------------------
		//層の情報を表示
		//------------------------------------------------------------------
		informationFormat("Layer Information");
		u32 index = 0;
		for (auto& layer : mLayerList)
		{
			layer->printLayerInfo();
			index++;
		}
		std::cout << std::endl;



		mAlreadyBuild = true;
	}


	void AI::deepLearning(f32* pTrainingData, f32* pCorrectData, u32 epochs)
	{
		//mOptimizer->setLearningRate(learningRate);

		//------------------------------------------------------------------
		//訓練データの情報を表示
		//------------------------------------------------------------------
		printTitle("Training Data Information");
#pragma region print_TrainingData_infomation
		auto printer = [](std::string name, u32 value, u32 stringLen = 25)
		{
			u32 res = stringLen - name.length();
#if _DEBUG
			if (res >= stringLen)
			{
				assert(0);
			}
#endif
			std::string space = std::string(res, ' ');
			std::cout << name << space << " = " << value << "\n";
		};
		printer("Total TrainingData num", mDataFormat4DeepLearning.dataNum);
		printer("channel", mDataFormat4DeepLearning.trainingDataShape.channel);
		printer("height", mDataFormat4DeepLearning.trainingDataShape.height);
		printer("width", mDataFormat4DeepLearning.trainingDataShape.width);
		std::cout << "\n";
		printer("channel", mDataFormat4DeepLearning.correctDataShape.channel);
		printer("height", mDataFormat4DeepLearning.correctDataShape.height);
		printer("width", mDataFormat4DeepLearning.correctDataShape.width);
		std::cout << std::endl;
#pragma endregion


		//------------------------------------------------------------------
		//順伝搬用のデータをここで準備する。
		//------------------------------------------------------------------
		dataSetup(pTrainingData, pCorrectData);


		//------------------------------------------------------------------
		//深層学習の実行箇所
		//------------------------------------------------------------------
		informationFormat("Deep Learning Start");
		u32 loopTime = mDataFormat4DeepLearning.dataNum / mDataFormat4DeepLearning.batchSize;
		u32 batch = mDataFormat4DeepLearning.batchSize;
		auto progressBar = [](u32 currentLoop, u32 totalLoop, f32 loss, u32 length = 80)
		{
			const u32 grid = totalLoop / length;
			std::string s = "\r";
			for (u32 i = 0; i < static_cast<u32>((static_cast<f32>(length) * currentLoop) / totalLoop); i++)
			{
				s += "=";
			}
			s += ">";
			const s32 spaceLength = static_cast<s32>(length - s.length() + 2);
			for (s32 i = 0; i < spaceLength; i++)
			{
				s += " ";
			}
			s += " " + std::to_string(static_cast<u32>(static_cast<f32>(currentLoop * 100) / totalLoop)) + "/100  :  " + std::to_string(loss);
			printf(s.c_str());
		};

		for (u32 epoch = 0; epoch < epochs; epoch++)
		{
			std::cout << "epoch = " << epoch + 1 << std::endl;
			f32 loss = 0.0f;
			std::cout << "deep learning now" << std::endl;
			for (u32 loop = 0; loop < loopTime; loop++)
			{
				progressBar(loop + 1, loopTime, mIsGpuAvailable ? mLossOnGPU : mLossOnCPU);
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

#if TIME_DEBUG
				{
					std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
					forward();
#if TIME_DEBUG
					f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
					std::string name = "";
					(name += __FUNCTION__) += " : forward";
					timers[name] = elapsedTime;
				}
#endif


				//------------------------------------------------------------------
				//ここで整合性チェック
				//------------------------------------------------------------------


#if TIME_DEBUG
				{
					std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
					backward();
#if TIME_DEBUG
					f32 elapsedTime = static_cast<f32>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f);
					std::string name = "";
					(name += __FUNCTION__) += " : backward";
					timers[name] = elapsedTime;
				}
#endif


				//------------------------------------------------------------------
				//ここで整合性チェック
				//------------------------------------------------------------------

#if TIME_DEBUG
				{
					std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
					optimize();
#if TIME_DEBUG
					f32 elapsedTime = static_cast<f32>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f);
					std::string name = "";
					(name += __FUNCTION__) += " : optimize";
					timers[name] = elapsedTime;
				}
#endif
				//------------------------------------------------------------------
				//ここで整合性チェック
				//------------------------------------------------------------------

#if TIME_DEBUG

				auto cmp = [](const auto& a, const auto& b) {return a.second < b.second; };
				std::vector<std::pair<std::string, f32> > sortedTimes;
				for (const auto& item : timers)
				{
					sortedTimes.emplace_back(item);
				}
				std::sort(sortedTimes.begin(), sortedTimes.end(), cmp);
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


	DataArray AI::operator()(f32* inputData)
	{
		if (!mAlreadyBuild)
		{
			std::cout << "AI build not yet done!!";
			assert(0);
		}
		return DataArray();
	}

#pragma endregion

#pragma region private

	/// <summary>
	/// GPUが使えるかAPIを使って調べる。
	/// CUDAを使えるGPUがある場合、それを全て列挙し、特性を出力する。
	/// </summary>
	void AI::checkGpuIsAvailable()
	{
		informationFormat("GPU Information");

		s32 gpuDeviceNum = 0;
		const cudaError_t error = cudaGetDeviceCount(&gpuDeviceNum);
		if (error != cudaSuccess)
		{
			std::cout << "Error : " << __FILE__ << ":" << __LINE__ << std::endl;
			std::cout << "code : " << error << std::endl;
			std::cout << "reason : " << cudaGetErrorString(error) << std::endl;
			std::cout << "\nSystem can't get any Device Information. So this AI uses CPU to DeepLearning.\n";
			mIsGpuAvailable = false;
			return;
		}

		if (gpuDeviceNum == 0)
		{
			std::cout << "No GPU Device that support CUDA. So this AI use CPU  to DeepLearning.";
			mIsGpuAvailable = false;
			return;
		}

		std::cout << gpuDeviceNum << " GPU device(s) that support CUDA detected.\n";

		s32 maxDeviceId = 0;
		s32 maxMultiProcessorCount = 0;

		for (s32 index = 0; index < gpuDeviceNum; index++)
		{
			cudaDeviceProp deviceProp;
			cudaGetDeviceProperties(&deviceProp, index);
			s32 driverVersion = 0;
			s32 runtimeVersion = 0;
			cudaDriverGetVersion(&driverVersion);
			cudaRuntimeGetVersion(&runtimeVersion);

			auto formater = [](std::string s, s32 length = 50)
			{
				return s + std::string(std::max(0, length - static_cast<s32>(s.length())), ' ') + " : ";
			};

			std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;
			std::cout << "Information of DeviceID = " << index << "\n" << std::endl;
			std::cout << formater("Device name") << "\"" << deviceProp.name << "\"" << std::endl;
			std::cout << formater("CUDA Driver Version") << driverVersion / 1000 << "." << (driverVersion % 100) / 10 << std::endl;
			std::cout << formater("CUDA Runtime Versionz") << runtimeVersion / 1000 << "." << (runtimeVersion % 100) / 10 << std::endl;
			std::cout << formater("CUDA Capability Major/Minor version number") << deviceProp.major << "." << deviceProp.minor << std::endl;

			std::cout << formater("VRAM") << static_cast<f32>(deviceProp.totalGlobalMem / pow(1024.0, 3)) << "GB (" << deviceProp.totalGlobalMem << "Bytes)" << std::endl;
			std::cout << formater("Total amount of shared memory per block") << deviceProp.sharedMemPerBlock << "Bytes" << std::endl;
			std::cout << formater("Max Texture Dimension Size of 1D") << "(" << deviceProp.maxTexture1D << ")" << std::endl;
			std::cout << formater("Max Texture Dimension Size of 2D") << "(" << deviceProp.maxTexture2D[0] << ", " << deviceProp.maxTexture2D[1] << ")" << std::endl;
			std::cout << formater("Max Texture Dimension Size of 3D") << "(" << deviceProp.maxTexture3D[0] << ", " << deviceProp.maxTexture3D[1] << ", " << deviceProp.maxTexture3D[2] << ")" << std::endl;
			std::cout << formater("Maximum sizes of threads per block") << deviceProp.maxThreadsPerBlock << std::endl;
			std::cout << formater("Maximum sizes of each dimension of a block") << "(" << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << ")" << std::endl;
			std::cout << formater("Maximum sizes of each dimension of a grid") << "(" << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << ")" << std::endl;



			if (deviceProp.multiProcessorCount > maxMultiProcessorCount)
			{
				maxMultiProcessorCount = deviceProp.multiProcessorCount;
				maxDeviceId = index;
			}
		}

		std::cout << "---------------------------------------------------------------------------------------------------" << std::endl;
		std::cout << "\n" << std::endl;
		std::cout << "this time AI use deviceID = " << maxDeviceId << std::endl;
		cudaSetDevice(maxDeviceId);
		mIsGpuAvailable = ON_CPU_DEBUG ? false : true;
		std::cout << std::endl;
	}


	/// <summary>
	///　各層におけるパラメータのためのメモリ確保や初期化、
	/// そして学習時に各層が必要となる前の層の出力データのアドレスを登録。
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


	void AI::initializeOptimizer()
	{
		if (mIsGpuAvailable)
		{
			mOptimizer->initializeOnGPU(mLayerList);
		}
		else
		{
			mOptimizer->initializeOnCPU(mLayerList);
		}
	}

	/// <summary>
	/// AIクラス利用者が指定してきたデータをデータフォーマットに応じ、
	/// またGPUの利用可否にも応じて、データの準備を行う。
	/// </summary>
	/// <param name="pTrainingData">訓練データのCPUアドレス</param>
	/// <param name="pCorrectData">教師データのCPUアドレス</param>
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
			mInputTrainingDataOnCPU.setSizeAs4D(batch, mDataFormat4DeepLearning.trainingDataShape);
			mInputTrainingDataOnCPU.byteSize = mInputTrainingDataOnCPU.size * sizeof(f32);
			mInputCorrectDataOnCPU.setSizeAs4D(batch, mDataFormat4DeepLearning.correctDataShape);
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