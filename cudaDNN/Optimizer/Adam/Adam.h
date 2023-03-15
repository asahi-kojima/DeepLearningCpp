#pragma once

#include "../BaseOptimizer.h"

namespace miduho
{
	namespace optimizer
	{
		class Adam : public BaseOptimizer
		{
			SINGLETON(Adam)

		private:
			void optimize(layer::BaseLayer*) override;
		};
	}
}