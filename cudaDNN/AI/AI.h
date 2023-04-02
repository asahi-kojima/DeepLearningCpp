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
		struct InputDataInterpretation
		{
			u32 totalDataNum;
			u32 elementNum;
			u32 toalElementNum;

			u32 byteSize;
			u32 totalByteSize;

			InputDataShape shape;

			InputDataInterpretation() = default;
			InputDataInterpretation(u32 totalDataNum, InputDataShape shape)
				: totalDataNum(totalDataNum)
				, shape{ shape }
				, elementNum(shape.channel* shape.height* shape.width)
				, toalElementNum(elementNum * totalDataNum)
				, byteSize(elementNum * sizeof(f32))
				, totalByteSize(totalDataNum* byteSize)
			{}
		};

		AI();
		~AI();

		void addLayer(std::unique_ptr<layer::BaseLayer>&&);
		void build(InputDataInterpretation&, std::unique_ptr<optimizer::BaseOptimizer>&&, std::unique_ptr<lossFunction::BaseLossFunction>&&);
		
		void deepLearning(f32*, f32*);
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


		//入力データの基点
		f32* mInputTrainingDataStartAddressOnCpu;
		f32* mInputTrainingLableStartAddressOnCpu;

		f32* mInputTrainingDataStartAddressOnGpu;
		f32* mInputTrainingLableStartAddressOnGpu;

		//入力データを置く場所
		DataMemory mInputTrainingData;
		DataMemory mInputLabelData;
		//順伝搬の結果がある場所
		DataMemory* mForwardResult;

		
#if _DEBUG //GPUデバッグのための変数
		DataMemory mInputTrainingDataForGpuDebug;
		DataMemory mInputLabelDataForGpuDebug;
		DataMemory* mForwardResultForGpuDebug;
#endif

		f32 mLoss;

		bool mIsGpuAvailable = true;
		InputDataInterpretation mInterpretation;
	};
}