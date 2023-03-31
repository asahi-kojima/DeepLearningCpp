#pragma once
#include <vector>
#include <cassert>

#include "../AISetting.h"
#include "../Optimizer/BaseOptimizer.h"

namespace Aoba
{
	namespace layer
	{
		class BaseLayer
		{
			friend class Aoba::optimizer::BaseOptimizer;

		public:
			struct DataShape
			{
				u32 batchSize;
				u32 channel;
				u32 height;
				u32 width;
			};


			BaseLayer() = default;
			virtual ~BaseLayer() = default;


			/// <summary>
			/// 入力・出力サイズ、カーネルサイズなどの決定などを行う。
			/// </summary>
			/// <param name=""></param>
			virtual void setupLayerInfo(DataShape*) = 0;

			/// <summary>
			/// メモリの確保などを行う。
			/// </summary>
			virtual void initialize() final
			{
				if (mIsGpuAvailable)
				{
					initializeOnGPU();
					if (_DEBUG)
						initializeOnCPU();
				}
				else
				{
					initializeOnCPU();
				}
			}
			virtual void forward() final
			{
				if (mIsGpuAvailable)
				{
					forwardOnGPU();
					if (_DEBUG)
						forwardOnCPU();
				}
				else
				{
					forwardOnCPU();
				}
			}
			virtual void backward() final
			{
				if (mIsGpuAvailable)
				{
					backwardOnGPU();
					if (_DEBUG)
					{
						backwardOnCPU();
					}
				}
				else
				{
					backwardOnCPU();
				}
			}
			virtual void terminate() final
			{
				if (mIsGpuAvailable)
				{
					terminateOnGPU();
					if (_DEBUG)
					{
						terminateOnCPU();
					}
				}
				else
				{
					terminateOnCPU();
				}
			}


			virtual void setInputData(DataMemory*& pInputData) final
			{
				if (mIsGpuAvailable)
				{
					mInputDataOnGPU = pInputData;
					pInputData = &mForwardResultOnGPU;
				}
				else
				{
					mInputDataOnCPU = pInputData;
					pInputData = &mForwardResultOnCPU;
				}
			}
			virtual void setDInputData(DataMemory*& pDInputData) final
			{
				if (mIsGpuAvailable)
				{
					mDInputDataOnGPU = pDInputData;
					pDInputData = &mBackwardResultOnGPU;
				}
				else
				{
					mDInputDataOnGPU = pDInputData;
					pDInputData = &mBackwardResultOnCPU;
				}
			}
#if _DEBUG
			virtual void setInputDataForGpuDebug(DataMemory*& pInputData) final
			{
				mInputDataOnCPU = pInputData;
				pInputData = &mForwardResultOnCPU;
			}
			virtual void setDInputDataForGpuDebug(DataMemory*& pDInputData) final
			{
				mDInputDataOnGPU = pDInputData;
				pDInputData = &mBackwardResultOnCPU;
			}
#endif

			virtual void setIsGpuAvailable(bool which) final
			{
				mIsGpuAvailable = which;
			}

			virtual void memcpyHostToDevice() = 0;
			virtual void memcpyDeviceToHost() = 0;

		protected:
			bool mIsGpuAvailable = false;

			//CPU関係の変数と関数
			std::vector<paramMemory> pParametersOnCPU;
			std::vector<paramMemory> pDParametersOnCPU;
			DataMemory mForwardResultOnCPU;
			DataMemory mBackwardResultOnCPU;
			DataMemory* mInputDataOnCPU;
			DataMemory* mDInputDataOnCPU;

			void setInputDataOnCPU(DataMemory* pInputDataOnCPU)
			{
				mInputDataOnCPU = pInputDataOnCPU;
			}
			void setDInputDataOnCPU(DataMemory* pDInputDataOnCPU)
			{
				mDInputDataOnCPU = pDInputDataOnCPU;
			}

			virtual void initializeOnGPU() = 0;
			virtual void forwardOnCPU() = 0;
			virtual void backwardOnCPU() = 0;
			virtual void terminateOnCPU() = 0;





			//GPU関係の変数と関数
			std::vector<paramMemory> pParametersOnGPU;
			std::vector<paramMemory> pDParametersOnGPU;
			DataMemory mForwardResultOnGPU;
			DataMemory mBackwardResultOnGPU;
			DataMemory* mInputDataOnGPU;
			DataMemory* mDInputDataOnGPU;

			void setInputDataOnGPU(DataMemory* pInputDataOnGPU)
			{
				mInputDataOnGPU = pInputDataOnGPU;
			}
			void setDInputDataOnGPU(DataMemory* pDInputDataOnGPU)
			{
				mDInputDataOnGPU = pDInputDataOnGPU;
			}

			virtual void initializeOnCPU() = 0;
			virtual void forwardOnGPU() = 0;
			virtual void backwardOnGPU() = 0;
			virtual void terminateOnGPU() = 0;





		};


	}
}