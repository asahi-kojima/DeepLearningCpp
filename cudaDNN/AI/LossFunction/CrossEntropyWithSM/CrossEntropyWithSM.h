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

			f32 calcLossAndDInputOnGPU() override;
			f32 calcLossAndDInputOnCPU() override;
		};
	}
}