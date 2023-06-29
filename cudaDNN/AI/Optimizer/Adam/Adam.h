#pragma once
#include <map>
#include "../BaseOptimizer.h"

namespace Aoba
{
	namespace optimizer
	{
		class Adam : public BaseOptimizer
		{
		public:
			Adam(f32 = 0.001f, f32 = 0.9,f32 = 0.9);
			~Adam() = default;

		
		private:
			void initializeOnCPU(std::vector<std::unique_ptr<layer::BaseLayer> >&) override;
			void initializeOnGPU(std::vector<std::unique_ptr<layer::BaseLayer> >&) override;
			void terminateOnCPU() override;
			void terminateOnGPU() override;
			void optimizeOnCPU(std::unique_ptr<layer::BaseLayer>&) override;
			void optimizeOnGPU(std::unique_ptr<layer::BaseLayer>&) override;

			std::vector<std::vector<DataArray> > mMomentumOnCPU;
			std::vector<std::vector<DataArray> > mVelocityOnCPU;
			std::map<layer::BaseLayer*, u32> mLayerOrderMapOnCPU;

			std::vector<std::vector<DataArray> > mMomentumOnGPU;
			std::vector<std::vector<DataArray> > mVelocityOnGPU;
			std::map<layer::BaseLayer*, u32> mLayerOrderMapOnGPU;

			f32 mBeta0;
			f32 mBeta1;

			u32 mIteration;
		};
	}
}