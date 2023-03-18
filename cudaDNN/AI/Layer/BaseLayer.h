#pragma once
#include <vector>
#include "../../setting.h"
#include "../Optimizer/BaseOptimizer.h"

namespace Aoba
{
	namespace layer
	{
		class BaseLayer
		{
			friend class Aoba::optimizer::BaseOptimizer;
		public:
			using parameterType = f32;
			using flowDataType = f32;

			struct DataShape
			{
				u32 batchSize;
				u32 channel;
				u32 height;
				u32 width;
			};

			struct DataMemory
			{
				flowDataType* address;
				u32 size;
			};

			using constDataMemory = const DataMemory;

			struct paramMemory
			{
				parameterType* address;
				u32 size;
			};


		public:
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
			virtual void initialize() = 0;
			virtual void forward() = 0;
			virtual void backward() = 0;
			virtual void terminate() = 0;

			

			constDataMemory* setInputData(constDataMemory* pInputData)
			{
#ifdef GPU_AVAILABLE
				mInputDataOnGPU = pInputData;
				return &mForwardResultOnGPU;
#else
				mInputDataOnCPU = pInputData;
				return &mForwardResultOnCPU;
#endif
			}
			constDataMemory* setDInputData(constDataMemory* pDInputData)
			{
#ifdef GPU_AVAILABLE
				mDInputDataOnGPU = pDInputData;
				return &mBackwardResultOnGPU;
#else
				mDInputDataOnGPU = pDInputData;
				return &mBackwardResultOnCPU;
#endif
			}

			virtual void memcpyHostToDevice() = 0;
			virtual void memcpyDeviceToHost() = 0;

		protected:

			bool isInitialized = false;
			
			
			//CPU関係の変数と関数
			std::vector<paramMemory> pParametersOnCPU;
			std::vector<paramMemory> pDParametersOnCPU;
			DataMemory mForwardResultOnCPU;
			DataMemory mBackwardResultOnCPU;
			constDataMemory* mInputDataOnCPU;
			constDataMemory* mDInputDataOnCPU;

			void setInputDataOnCPU(constDataMemory* pInputDataOnCPU)
			{
				mInputDataOnCPU = pInputDataOnCPU;
			}
			void setDInputDataOnCPU(constDataMemory* pDInputDataOnCPU)
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
			constDataMemory* mInputDataOnGPU;
			constDataMemory* mDInputDataOnGPU;

			void setInputDataOnGPU(constDataMemory* pInputDataOnGPU)
			{
				mInputDataOnGPU = pInputDataOnGPU;
			}
			void setDInputDataOnGPU(constDataMemory* pDInputDataOnGPU)
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