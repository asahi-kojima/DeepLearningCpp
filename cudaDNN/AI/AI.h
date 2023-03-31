#pragma once
#include <vector>
#include <memory>

#include "AISetting.h"
#include "./Layer/BaseLayer.h"
#include "./Optimizer/BaseOptimizer.h"
#include "./LossFunction/BaseLossFunction.h"

#define CREATELAYER(classname, ...) std::make_unique<classname>(__VA_ARGS__)
namespace Aoba
{
	class AI
	{
	public:
		using InputDataShape = layer::BaseLayer::DataShape;
		struct InputData
		{
			cf32* address;
			u32 totalDataNum;
			u32 channel;
			u32 height;
			u32 width;

			u32 dataSize;
			u32 byteSize;
			u32 totalByteSize;
			InputData(f32* address, u32 totalDataNum, u32 channel, u32 height, u32 width)
				: address(address)
				, totalDataNum(totalDataNum)
				, channel(channel)
				, height(height)
				, width(width)
			{
				dataSize = channel * height * width;
				assert(totalDataNum % dataSize == 0);
				byteSize = dataSize * sizeof(f32);
				totalByteSize = 0;
			}
		};

		AI();
		~AI();

		void addLayer(std::unique_ptr<layer::BaseLayer>&&);
		void build(InputDataShape&, std::unique_ptr<optimizer::BaseOptimizer>&&, std::unique_ptr<lossFunction::BaseLossFunction>&&);
		
		void setLearningData();
		void deepLearning();
		DataMemory operator()(f32*);//未実装
		f32 getLoss() { return mLoss; }


	private:
		void checkGpuIsAvailable();
		void setupLayerInfo(InputDataShape&);
		void allocLayerMemory();

		void forward();
		void backward();
		void optimize();

		//AIを構成する層のリスト
		std::vector<std::unique_ptr<layer::BaseLayer> > mLayerList;
		//オプティマイザー
		std::unique_ptr<optimizer::BaseOptimizer> mOptimizer;
		//損失関数
		std::unique_ptr<lossFunction::BaseLossFunction> mLossFunction;


		//入力データを置く場所
		DataMemory mInputTrainingData;
		DataMemory mInputLabelData;
		//順伝搬の結果がある場所
		DataMemory* mForwardResult;
		//逆伝搬のための入力データ
		DataMemory mDInputData;

		
#if _DEBUG //GPUデバッグのための変数
		DataMemory mInputTrainingDataForGpuDebug;
		DataMemory mInputTestDataForGpuDebug;
		DataMemory* mForwardResultForGpuDebug;
		DataMemory mDInputDataForGpuDebug;
#endif

		f32 mLoss;

		bool mIsGpuAvailable = true;
	};
}