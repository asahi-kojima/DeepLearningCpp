#pragma once

#include "../BaseLossFunction.h"

namespace Aoba
{
	namespace lossFunction
	{
		class CrossEntropyWithSM : public BaseLossFunction
		{
		public:
			CrossEntropyWithSM() = default;
			~CrossEntropyWithSM() = default;

		private:
			void initializeOnGPU() override;
			void initializeOnCPU() override;

			f32 calcLossAndDInputOnGPU(constDataMemory&, void*) override;
			f32 calcLossAndDInputOnCPU(constDataMemory&, void*) override;
		};
	}
}