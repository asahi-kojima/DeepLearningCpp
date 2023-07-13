#pragma once
#include <vector>
#include <cassert>
#include <iostream>
#include  "../AIHelperFunction.h"
#include "../AIDataStructure.h"
#include "../AIMacro.h"
#include "../Optimizer/BaseOptimizer.h"

namespace Aoba
{
	namespace layer
	{
		class BaseLayer
		{
			friend class Aoba::optimizer::BaseOptimizer;

		public:
			BaseLayer() = default;
			virtual ~BaseLayer() = default;

			virtual void setInputDataOnCPU(DataArray*& pInputData) final
			{
				mInputDataOnCPU = pInputData;
				pInputData = &mForwardResultOnCPU;
			}
			virtual void setDInputDataOnCPU(DataArray*& pDInputData) final
			{
				mDInputDataOnCPU = pDInputData;
				pDInputData = &mBackwardResultOnCPU;
			}

			virtual void setInputDataOnGPU(DataArray*& pInputData) final
			{
				mInputDataOnGPU = pInputData;
				pInputData = &mForwardResultOnGPU;
			}
			virtual void setDInputDataOnGPU(DataArray*& pDInputData) final
			{
				mDInputDataOnGPU = pDInputData;
				pDInputData = &mBackwardResultOnGPU;
			}

			virtual void printLayerInfo()
			{
				printDoubleLine();
				std::cout << "No Infomation" << std::endl;
			}

			u32 getTotalParameterNum(bool isGpuUse) const
			{
				u32 parameterNum = 0;
				if (isGpuUse)
				{
					for (auto& parameters : mParametersPtrOnGPU)
					{
						parameterNum += parameters.size;
					}
				}
				else
				{
					for (auto& parameters : mParametersPtrOnCPU)
					{
						parameterNum += parameters.size;
					}
				}
			}


			template<typename T, typename ... Args>
			static void unitTest(DataShape inputShape, Args ... args)
			{
				std::unique_ptr<BaseLayer> pLayer = std::make_unique<T>(args...);

				const f32 AllowableError = 1e-2;// 1パーセント

				//層の初期化を行う(層情報の決定とメモリ確保)
				DataShape shape4CPU = inputShape;
				DataShape shape4GPU = inputShape;
				pLayer->initializeOnCPU(10, shape4CPU);
				pLayer->initializeOnGPU(10, shape4GPU);

				//入力データをセットしている。
				DataArray inputDataOnCPU;
				DataArray inputDataOnGPU;
				DataArray* pInputDataOnCPU = &inputDataOnCPU;
				DataArray* pInputDataOnGPU = &inputDataOnGPU;
				pLayer->setInputDataOnCPU(pInputDataOnCPU);
				pLayer->setInputDataOnGPU(pInputDataOnGPU);

				//パラメータのコピー（乱数の場合がある為）
				for (u32 i = 0; i < pLayer->mDParametersPtrOnCPU.size(); i++)
				{
					CHECK(cudaMemcpy(
						pLayer->mParametersPtrOnGPU[i].address,
						pLayer->mParametersPtrOnCPU[i].address,
						pLayer->mParametersPtrOnCPU[i].size * sizeof(f32),
						cudaMemcpyHostToDevice));
				}

				
				{
					//インプットデータの生成
					inputDataOnCPU.setSizeAs4D(10, inputShape);
					inputDataOnGPU.setSizeAs4D(10, inputShape);
					MALLOC_AND_INITIALIZE_NORMAL_ON_CPU(inputDataOnCPU, 10, 1.0f);
					MALLOC_ON_GPU(inputDataOnGPU);
					CHECK(cudaMemcpy(inputDataOnGPU.address, inputDataOnCPU.address, inputDataOnCPU.size * sizeof(f32), cudaMemcpyHostToDevice));

					//順伝搬
					pLayer->forwardOnCPU();
					pLayer->forwardOnGPU();
					
					f32* gpuResult = new f32[pInputDataOnCPU->size];
					CHECK(cudaMemcpy(
						gpuResult,
						pInputDataOnGPU->address,
						pInputDataOnGPU->size * sizeof(f32),
						cudaMemcpyDeviceToHost));

					for (u32 i = 0; i < pInputDataOnCPU->size; i++)
					{
						f32 x = gpuResult[i];
						f32 x0 = (*pInputDataOnCPU)[i];

						f32 error;
						if (abs(x0) < 1e-7)
						{
							error = abs(x);
						}
						else
						{
							error = abs((x - x0) / x0);
						}

						if (error > AllowableError)
						{
							assert(0);
						}
					}
					delete[] gpuResult;

					std::cout << "forward check clear\n";
				}

				//逆伝搬データをセットしている。
				DataArray dInputDataOnCPU;
				DataArray dInputDataOnGPU;
				DataArray* pDInputDataOnCPU = &dInputDataOnCPU;
				DataArray* pDInputDataOnGPU = &dInputDataOnGPU;
				pLayer->setDInputDataOnCPU(pDInputDataOnCPU);
				pLayer->setDInputDataOnGPU(pDInputDataOnGPU);

				{
					//インプットデータの生成
					dInputDataOnCPU.setSizeAs4D(10, shape4CPU);
					dInputDataOnGPU.setSizeAs4D(10, shape4GPU);
					MALLOC_AND_INITIALIZE_NORMAL_ON_CPU(dInputDataOnCPU, 10, 1.0f);
					MALLOC_ON_GPU(dInputDataOnGPU);
					CHECK(cudaMemcpy(dInputDataOnGPU.address, dInputDataOnCPU.address, dInputDataOnCPU.size * sizeof(f32), cudaMemcpyHostToDevice));

					//g逆伝搬
					pLayer->backwardOnCPU();
					pLayer->backwardOnGPU();
					//逆伝搬結果のチェック
					{
						f32* gpuResult = new f32[pDInputDataOnCPU->size];
						CHECK(cudaMemcpy(
							gpuResult,
							pDInputDataOnGPU->address,
							pDInputDataOnGPU->size * sizeof(f32),
							cudaMemcpyDeviceToHost));

						for (u32 i = 0; i < pDInputDataOnCPU->size; i++)
						{
							f32 x = gpuResult[i];
							f32 x0 = (*pDInputDataOnCPU)[i];

							f32 error;
							if (abs(x0) < 1e-7)
							{
								error = abs(x);
							}
							else
							{
								error = abs((x - x0) / x0);
							}

							if (error > AllowableError)
							{
								assert(0);
							}
						}
						delete[] gpuResult;

						std::cout << "backward check clear\n";
					}
				}


				//逆伝搬パラメータのチェック
				{
					for (u32 i = 0; i < pLayer->mDParametersPtrOnCPU.size(); i++)
					{
						f32* gpuResult = new f32[pLayer->mDParametersPtrOnGPU[i].size];
						CHECK(cudaMemcpy(
							gpuResult,
							pLayer->mDParametersPtrOnGPU[i].address,
							pLayer->mDParametersPtrOnGPU[i].size * sizeof(f32),
							cudaMemcpyDeviceToHost));

						for (u32 j = 0; j < pLayer->mDParametersPtrOnGPU[i].size; j++)
						{
							f32 x = gpuResult[j];
							f32 x0 = (pLayer->mDParametersPtrOnCPU[i])[j];

							f32 error;
							if (abs(x0) < 1e-7)
							{
								error = abs(x);
							}
							else
							{
								error = abs((x - x0) / x0);
							}

							if (error > AllowableError)
							{
								assert(0);
							}
						}
						delete[] gpuResult;

						std::cout << "parameter check clear\n";
					}
				}
			}

			//CPU
			std::vector<DataArray> mParametersPtrOnCPU;
			std::vector<DataArray> mDParametersPtrOnCPU;
			DataArray mForwardResultOnCPU;
			DataArray mBackwardResultOnCPU;
			DataArray* mInputDataOnCPU;
			DataArray* mDInputDataOnCPU;

			virtual void initializeOnCPU(u32 batchSize, DataShape& shape) final
			{
				if (!mIsSetupLayerInfo)
				{
					setupLayerInfo(batchSize, shape);
					mIsSetupLayerInfo = true;
				}
				mallocOnCPU();
			}
			virtual void mallocOnCPU() = 0;

			virtual void forwardOnCPU() = 0;
			virtual void backwardOnCPU() = 0;
			virtual void terminateOnCPU() = 0;


			//GPU
			std::vector<DataArray> mParametersPtrOnGPU;
			std::vector<DataArray> mDParametersPtrOnGPU;
			DataArray mForwardResultOnGPU;
			DataArray mBackwardResultOnGPU;
			DataArray* mInputDataOnGPU;
			DataArray* mDInputDataOnGPU;

			virtual void initializeOnGPU(u32 batchSize, DataShape& shape) final
			{
				if (!mIsSetupLayerInfo)
				{
					setupLayerInfo(batchSize, shape);
					mIsSetupLayerInfo = true;
				}
				mallocOnGPU();
			}
			virtual void mallocOnGPU() = 0;

			virtual void forwardOnGPU() = 0;
			virtual void backwardOnGPU() = 0;
			virtual void terminateOnGPU() = 0;

		private:
			virtual void setupLayerInfo(u32, DataShape&) = 0;
			bool mIsSetupLayerInfo = false;
		};
	}
}