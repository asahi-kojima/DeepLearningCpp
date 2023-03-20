#pragma once

#include "../AISetting.h"
#include "../Layer/BaseLayer.h"
namespace Aoba
{
	namespace lossFunction
	{
		class BaseLossFunction
		{
		public:
			BaseLossFunction() = default;
			~BaseLossFunction() = default;

			void setupDataShape(layer::BaseLayer::DataShape& dataShape) { mDataShape = dataShape; }
			void initialize()
			{
#ifdef  GPU_AVAILABLE
				initializeOnGPU();
#else
				initializeOnCPU();
#endif //  GPU_AVAILABLE
			}

			f32 calcLossAndDInput(constDataMemory& data, void* label)
			{
#ifdef  GPU_AVAILABLE
				return calcLossAndDInputOnGPU(data, label);
#else
				return calcLossAndDInputOnCPU(data, label);
#endif //  GPU_AVAILABLE
			}
			DataMemory mDInputData;

		protected:
			layer::BaseLayer::DataShape mDataShape;
			virtual void initializeOnGPU()=0;
			virtual void initializeOnCPU()=0;

			virtual f32 calcLossAndDInputOnGPU(constDataMemory&, void*) = 0;
			virtual f32 calcLossAndDInputOnCPU(constDataMemory&, void*) = 0;
		};
	}
}