#pragma once
#include <vector>
#include <memory>

#include "../setting.h"
#include "./Layer/BaseLayer.h"
#include "./Optimizer/BaseOptimizer.h"

#define CREATELAYER(classname, ...) std::make_unique<classname>(__VA_ARGS__)
namespace Aoba
{
	class AI
	{
	public:
		using dataMemory = layer::BaseLayer::DataMemory;
		using constDataMemory = layer::BaseLayer::constDataMemory;
		using InputDataShape = layer::BaseLayer::DataShape;
		
		AI();
		~AI();

		void addLayer(std::unique_ptr<layer::BaseLayer>&&);
		void build(InputDataShape&, std::unique_ptr<optimizer::BaseOptimizer>&&);
		constDataMemory forward(f32*);
		constDataMemory backward(f32*);
		void optimize();

	private:
		using flowDataType = layer::BaseLayer::flowDataType;

		


		void setupLayerInfo(InputDataShape&);
		void allocLayerMemory();



		std::vector<std::unique_ptr<layer::BaseLayer> > mLayerList;
		std::unique_ptr<optimizer::BaseOptimizer> mOptimizer;
		dataMemory mInputData;
		constDataMemory* mForwardResult;
	};
}