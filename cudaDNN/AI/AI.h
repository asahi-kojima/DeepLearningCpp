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


		//���̓f�[�^�̊�_
		f32* mInputTrainingDataStartAddressOnCpu;
		f32* mInputTrainingLableStartAddressOnCpu;

		f32* mInputTrainingDataStartAddressOnGpu;
		f32* mInputTrainingLableStartAddressOnGpu;

		//���̓f�[�^��u���ꏊ
		DataMemory mInputTrainingData;
		DataMemory mInputLabelData;
		//���`���̌��ʂ�����ꏊ
		DataMemory* mForwardResult;

		
#if _DEBUG //GPU�f�o�b�O�̂��߂̕ϐ�
		DataMemory mInputTrainingDataForGpuDebug;
		DataMemory mInputLabelDataForGpuDebug;
		DataMemory* mForwardResultForGpuDebug;
#endif

		f32 mLoss;

		bool mIsGpuAvailable = true;
		InputDataInterpretation mInterpretation;
	};
}