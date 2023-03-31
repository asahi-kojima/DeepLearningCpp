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
		DataMemory operator()(f32*);//������
		f32 getLoss() { return mLoss; }


	private:
		void checkGpuIsAvailable();
		void setupLayerInfo(InputDataShape&);
		void allocLayerMemory();

		void forward();
		void backward();
		void optimize();

		//AI���\������w�̃��X�g
		std::vector<std::unique_ptr<layer::BaseLayer> > mLayerList;
		//�I�v�e�B�}�C�U�[
		std::unique_ptr<optimizer::BaseOptimizer> mOptimizer;
		//�����֐�
		std::unique_ptr<lossFunction::BaseLossFunction> mLossFunction;


		//���̓f�[�^��u���ꏊ
		DataMemory mInputTrainingData;
		DataMemory mInputLabelData;
		//���`���̌��ʂ�����ꏊ
		DataMemory* mForwardResult;
		//�t�`���̂��߂̓��̓f�[�^
		DataMemory mDInputData;

		
#if _DEBUG //GPU�f�o�b�O�̂��߂̕ϐ�
		DataMemory mInputTrainingDataForGpuDebug;
		DataMemory mInputTestDataForGpuDebug;
		DataMemory* mForwardResultForGpuDebug;
		DataMemory mDInputDataForGpuDebug;
#endif

		f32 mLoss;

		bool mIsGpuAvailable = true;
	};
}