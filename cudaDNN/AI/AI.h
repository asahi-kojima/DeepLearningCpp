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
		DataMemory operator()(f32*);//������
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

		void dataSetup(f32*, f32*);

		void forward();
		void backward();
		void optimize();

		//////////////////////////////////////////////////////////////////////
		//���ʕϐ�
		//////////////////////////////////////////////////////////////////////
		//AI���\������w�̃��X�g
		std::vector<std::unique_ptr<layer::BaseLayer> > mLayerList;
		//�I�v�e�B�}�C�U�[
		std::unique_ptr<optimizer::BaseOptimizer> mOptimizer;
		//�����֐�
		std::unique_ptr<lossFunction::BaseLossFunction> mLossFunction;

		bool mIsGpuAvailable = true;
		DataFormat4DeepLearning mDataFormat4DeepLearning;




		//////////////////////////////////////////////////////////////////////
		//CPU�֌W�̕ϐ�
		//////////////////////////////////////////////////////////////////////
		//���̓f�[�^�̊�_
		f32* mInputTrainingDataStartAddressOnCPU;
		f32* mInputCorrectDataStartAddressOnCPU;
		//���̓f�[�^��u���ꏊ
		DataMemory mInputTrainingDataOnCPU;
		DataMemory mInputCorrectDataOnCPU;
		//���`���̌��ʂ�����ꏊ
		DataMemory* mForwardResultOnCPU;

		f32 mLossOnCPU = 0.0f;


		
		//////////////////////////////////////////////////////////////////////
		//GPU�֌W�̕ϐ�
		//////////////////////////////////////////////////////////////////////
		//���̓f�[�^�̊�_
		f32* mInputTrainingDataStartAddressOnGPU;
		f32* mInputCorrectDataStartAddressOnGPU;
		//���̓f�[�^��u���ꏊ
		DataMemory mInputTrainingDataOnGPU;
		DataMemory mInputCorrectDataOnGPU;
		//���`���̌��ʂ�����ꏊ
		DataMemory* mForwardResultOnGPU;

		f32 mLossOnGPU = 0.0f;



	};
}