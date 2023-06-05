#pragma once
#include <vector>
#include <memory>

#include "AISetting.h"
#include "./Layer/BaseLayer.h"
#include "./Layer/Layer.h"
#include "./Optimizer/BaseOptimizer.h"
#include "./Optimizer/Optimizer.h"
#include "./LossFunction/BaseLossFunction.h"
#include "./LossFunction/LossFunction.h"

namespace Aoba
{


	class AI
	{
	public:
		using InputDataShape = DataShape;
		

		AI();
		~AI();

		template <typename T, typename ... Args>
		void addLayer(Args ... args)
		{
			mLayerList.push_back(std::make_unique<T>(args...));
		}

		template <typename T, typename ... Args>
		void setOptimizer(Args ... args)
		{
			mOptimizer = std::make_unique<T>(args...);
		}

		template <typename T, typename ... Args>
		void setLossFunction(Args ... args)
		{
			mLossFunction = std::make_unique<T>(args...);
		}

		void build(DataFormat4DeepLearning&);

		void deepLearning(f32*, f32*, u32 epochs = 50);
		DataArray operator()(f32*);//未実装
		f32 getLoss()
		{
			if (mIsGpuAvailable)
				return mLossOnGPU;
			else
				return mLossOnCPU;
		}


	private:
		void checkGpuIsAvailable();
		void initializeLayer();
		void initializeOptimizer();

		void dataSetup(f32*, f32*);

		void forward();
		void backward();
		void optimize();

		//////////////////////////////////////////////////////////////////////
		//共通変数
		//////////////////////////////////////////////////////////////////////
		//AIを構成する層のリスト
		std::vector<std::unique_ptr<layer::BaseLayer> > mLayerList;
		//オプティマイザー
		std::unique_ptr<optimizer::BaseOptimizer> mOptimizer;
		//損失関数
		std::unique_ptr<lossFunction::BaseLossFunction> mLossFunction;

		bool mIsGpuAvailable = false;
		DataFormat4DeepLearning mDataFormat4DeepLearning;

		bool mAlreadyBuild = false;




		//////////////////////////////////////////////////////////////////////
		//CPU関係の変数
		//////////////////////////////////////////////////////////////////////
		//入力データの基点
		f32* mInputTrainingDataStartAddressOnCPU;
		f32* mInputCorrectDataStartAddressOnCPU;
		//入力データを置く場所
		DataArray mInputTrainingDataOnCPU;
		DataArray mInputCorrectDataOnCPU;
		//順伝搬の結果がある場所
		DataArray* mForwardResultOnCPU;

		f32 mLossOnCPU = 0.0f;


		
		//////////////////////////////////////////////////////////////////////
		//GPU関係の変数
		//////////////////////////////////////////////////////////////////////
		//入力データの基点
		f32* mInputTrainingDataStartAddressOnGPU;
		f32* mInputCorrectDataStartAddressOnGPU;
		//入力データを置く場所
		DataArray mInputTrainingDataOnGPU;
		DataArray mInputCorrectDataOnGPU;
		//順伝搬の結果がある場所
		DataArray* mForwardResultOnGPU;

		f32 mLossOnGPU = 0.0f;



	};
}