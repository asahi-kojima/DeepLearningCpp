#pragma once
#include <vector>
#include "../setting.h"

#define virtualFuncOnCPUGPU(funcName)		\
{											\
public:										\
	virtual funcName() = 0;					\
private:									\
	virtual funcName##OnCPU() = 0;			\
	virtual funcName##OnCPU() = 0;			\
}

namespace miduho
{
	namespace layer
	{
		class BaseLayer
		{
		public:
			using parameterType = f32;
			using flowDataType = f32;

			struct FlowDataFormat
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
			virtual void setupLayerInfo(FlowDataFormat*) = 0;

			/// <summary>
			/// メモリの確保などを行う。
			/// </summary>
			virtual void initialize() = 0;
			virtual void forward() = 0;
			virtual void backward() = 0;
			virtual void terminate() = 0;

			
			DataMemory* getDataMemory()
			{
#ifdef GPU_AVAILABLE
				return &mForwardResultOnGPU;
#else
				return &mForwardResultOnCPU;
#endif
			}
			DataMemory* getDDataMemory()
			{
#ifdef GPU_AVAILABLE
				return &mBackwardResultOnGPU;
#else
				return &mBackwardResultOnCPU;
#endif
			}
			void setInputData(DataMemory* pInputData)
			{
#ifdef GPU_AVAILABLE
				mInputDataOnGPU = pInputData;
#else
				mInputDataOnCPU = pInputData;
#endif
			}
			void setDInputData(DataMemory* pDInputData)
			{
#ifdef GPU_AVAILABLE
				mDInputDataOnGPU = pDInputData;
#else
				mDInputDataOnGPU = pDInputData;
#endif
			}

			virtual void memcpyHostToDevice() = 0;
			virtual void memcpyDeviceToHost() = 0;

		protected:

			bool isInitialized = false;
			
			
			//CPU周りの変数
			std::vector<paramMemory> pParametersOnCPU;
			std::vector<paramMemory> pDParametersOnCPU;
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


			DataMemory* mInputDataOnCPU;
			DataMemory mForwardResultOnCPU;
			DataMemory* mDInputDataOnCPU;
			DataMemory mBackwardResultOnCPU;
			


			//GPU周りの変数
			std::vector<paramMemory> pParametersOnGPU;
			std::vector<paramMemory> pDParametersOnGPU;
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

			DataMemory* mInputDataOnGPU;
			DataMemory mForwardResultOnGPU;
			DataMemory* mDInputDataOnGPU;
			DataMemory mBackwardResultOnGPU;




		};

		
	}
}