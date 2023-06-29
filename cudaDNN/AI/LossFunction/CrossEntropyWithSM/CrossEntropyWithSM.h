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
			void mallocOnGPU() override;
			void mallocOnCPU() override;

			f32 calcLossAndDInputOnGPU() override;
			f32 calcLossAndDInputOnCPU() override;

			void terminateOnGPU() override;
			void terminateOnCPU() override;
		};
	}
}