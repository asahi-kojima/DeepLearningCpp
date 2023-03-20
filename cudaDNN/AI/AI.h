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


		//AI���\������w�̃��X�g
		std::vector<std::unique_ptr<layer::BaseLayer> > mLayerList;
		//�I�v�e�B�}�C�U�[
		std::unique_ptr<optimizer::BaseOptimizer> mOptimizer;
		//�����֐�
		std::unique_ptr<lossFunction::BaseLossFunction> mLossFunction;


		//���̓f�[�^��u���ꏊ
		DataMemory mInputData;
		//���`���̌��ʂ�����ꏊ
		constDataMemory* mForwardResult;
		//�t�`���̂��߂̓��̓f�[�^
		DataMemory mDInputData;

		f32 mLoss;
	};
}