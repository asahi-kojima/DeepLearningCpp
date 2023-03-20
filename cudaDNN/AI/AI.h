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
		
		AI();
		~AI();

		void addLayer(std::unique_ptr<layer::BaseLayer>&&);
		void build(InputDataShape&, std::unique_ptr<optimizer::BaseOptimizer>&&, std::unique_ptr<lossFunction::BaseLossFunction>&&);


		constDataMemory forward(f32*, void*);
		void backward();
		void optimize();
		f32 getLoss() { return mLoss; }

	private:
		void setupLayerInfo(InputDataShape&);
		void allocLayerMemory();


		//AIを構成する層のリスト
		std::vector<std::unique_ptr<layer::BaseLayer> > mLayerList;
		//オプティマイザー
		std::unique_ptr<optimizer::BaseOptimizer> mOptimizer;
		//損失関数
		std::unique_ptr<lossFunction::BaseLossFunction> mLossFunction;


		//入力データを置く場所
		DataMemory mInputData;
		//順伝搬の結果がある場所
		constDataMemory* mForwardResult;
		//逆伝搬のための入力データ
		DataMemory mDInputData;

		f32 mLoss;
	};
}