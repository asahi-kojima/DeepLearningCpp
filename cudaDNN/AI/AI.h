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
		using InputDataShape = DataShape;
		struct DataFormat4DeepLearning
		{
			u32 dataNum;
			u32 batchSize = 100;

			DataShape trainingDataShape;
			u32 eachTrainingDataSize;

			DataShape correctDataShape;
			u32 eachCorrectDataSize;

			DataFormat4DeepLearning(u32 dataNum,DataShape trainingDataShape, DataShape correctDataShape)
				: dataNum(dataNum)
				, trainingDataShape(trainingDataShape)
				, correctDataShape(correctDataShape)
			{
				if (trainingDataShape.batchSize != correctDataShape.batchSize)
				{
					correctDataShape.batchSize = trainingDataShape.batchSize;
				}

				eachTrainingDataSize = trainingDataShape.channel * trainingDataShape.height * trainingDataShape.width;
				eachCorrectDataSize = correctDataShape.channel * correctDataShape.height * correctDataShape.width;
			}

			DataFormat4DeepLearning() = default;
		};

		AI();
		~AI();

		template <typename T, typename ... Args>
		void addLayer(Args ... args)
		{
			mLayerList.push_back(std::make_unique<T>(args...));
		}
		void build(DataFormat4DeepLearning&, std::unique_ptr<optimizer::BaseOptimizer>&&, std::unique_ptr<lossFunction::BaseLossFunction>&&);

		void deepLearning(f32*, f32*, u32 epochs = 50, f32 = 0.001f);
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
		void setupLayerInfo(InputDataShape&);
		void allocLayerMemory();

		void dataSetup(f32*, f32*);
		void forward();
		void backward();
		void optimize();

		//AI���\������w�̃��X�g
		std::vector<std::unique_ptr<layer::BaseLayer> > mLayerList;
		//�I�v�e�B�}�C�U�[
		std::unique_ptr<optimizer::BaseOptimizer> mOptimizer;
		//�����֐�
		std::unique_ptr<lossFunction::BaseLossFunction> mLossFunction;


		//���̓f�[�^�̊�_
		f32* mInputTrainingDataStartAddressOnCPU;
		f32* mInputTrainingLableStartAddressOnCPU;
		//���̓f�[�^��u���ꏊ
		DataMemory mInputTrainingDataOnCPU;
		DataMemory mInputLabelDataOnCPU;
		//���`���̌��ʂ�����ꏊ
		DataMemory* mForwardResultOnCPU;

		f32 mLossOnCPU;



		//���̓f�[�^�̊�_
		f32* mInputTrainingDataStartAddressOnGPU;
		f32* mInputTrainingLableStartAddressOnGPU;
		//���̓f�[�^��u���ꏊ
		DataMemory mInputTrainingDataOnGPU;
		DataMemory mInputLabelDataOnGPU;
		//���`���̌��ʂ�����ꏊ
		DataMemory* mForwardResultOnGPU;

		f32 mLossOnGPU;



		bool mIsGpuAvailable = true;
		DataFormat4DeepLearning mDataFormat4DeepLearning;
	};
}