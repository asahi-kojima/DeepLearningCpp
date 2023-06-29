#pragma once
#include "../BaseLossFunction.h"

namespace Aoba
{
	namespace lossFunction
	{
		class L2Loss : public BaseLossFunction
		{
		public:
			L2Loss() = default;
			~L2Loss() = default;

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