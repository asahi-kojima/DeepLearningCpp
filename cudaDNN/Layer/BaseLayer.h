#pragma once
#include <vector>
#include "../setting.h"

namespace miduho
{
	namespace layer
	{
		class BaseLayer
		{
		public:
			using parameterType = f32;
			using flowDataType = f32;

			struct flowDataFormat
			{
				u32 batchSize;
				u32 channel;
				u32 height;
				u32 width;
			};

			struct dataMemory
			{
				flowDataType* dataAddress;
				u32 dataNum;
			};

		public:
			BaseLayer() = default;
			virtual ~BaseLayer() = default;

			/// <summary>
			/// 入力・出力サイズ、カーネルサイズなどの決定などを行う。
			/// </summary>
			/// <param name=""></param>
			virtual void initialize(flowDataFormat*) = 0;

			/// <summary>
			/// メモリの確保などを行う。
			/// </summary>
			virtual void setup() = 0;


			virtual void terminate() = 0;
			virtual void forward() = 0;
			virtual void backward() = 0;
			virtual void memcpyHostToDevice() = 0;
			virtual void memcpyDeviceToHost() = 0;
			
			void setInputDataOnGPU(dataMemory* pInputDataOnGPU)
			{
				mInputDataOnGPU = pInputDataOnGPU;
			}
			void setDInputDataOnGPU(dataMemory* pDInputDataOnGPU)
			{
				mDInputDataOnGPU = pDInputDataOnGPU;
			}
			dataMemory* getDataMemory()
			{
				return &mForwardResultOnGPU;
			}
			
			dataMemory* getDDataMemory()
			{
				return &mBackwardResultOnGPU;
			}

		protected:
			struct paramMemory
			{
				parameterType* paramAddress;
				u32 paramNum;
			};
			
			
			virtual void forwardOnGPU() = 0;
			virtual void forwardOnCPU() = 0;
			virtual void backwardOnGPU() = 0;
			virtual void backwardOnCPU() = 0;

			std::vector<paramMemory> pParametersOnCPU;
			std::vector<paramMemory> pDParametersOnCPU;


			//GPU周りの変数
			std::vector<paramMemory> pParametersOnGPU;
			std::vector<paramMemory> pDParametersOnGPU;
			dataMemory* mInputDataOnGPU;
			dataMemory mForwardResultOnGPU;
			dataMemory* mDInputDataOnGPU;
			dataMemory mBackwardResultOnGPU;

			bool isInitialized = false;


		};

		
	}
}