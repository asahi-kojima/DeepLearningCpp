#include "Adam.h"
#include "../../Layer/Layer.h"
#include "../../AISetting.h"

namespace Aoba::optimizer
{
	Adam::Adam(f32 learningRate, f32 beta0, f32 beta1)
		:BaseOptimizer(learningRate)
		,mBeta0(beta0)
		,mBeta1(beta1)
		,mIteration(0)
	{
	}

	void Adam::initializeOnCPU(std::vector<std::unique_ptr<layer::BaseLayer> >& layerList)
	{
		u32 layerSize = layerList.size();

		mMomentumOnCPU.resize(layerSize);
		mVelocityOnCPU.resize(layerSize);

		for (u32 order = 0; order < layerSize; order++)
		{
			auto pLayer = layerList[order].get();
			mLayerOrderMapOnCPU[pLayer] = order;

			std::vector<DataArray>& parameters = getLayerParamOnCPU(layerList[order]);
			mMomentumOnCPU[order].resize(parameters.size());
			mVelocityOnCPU[order].resize(parameters.size());

			for (u32 id = 0; id < mMomentumOnCPU[order].size(); id++)
			{
				mMomentumOnCPU[order][id].size = parameters[id].size;
				mVelocityOnCPU[order][id].size = parameters[id].size;
				MALLOC_AND_INITIALIZE_0_ON_CPU(mMomentumOnCPU[order][id]);
				MALLOC_AND_INITIALIZE_0_ON_CPU(mVelocityOnCPU[order][id]);
			}
		}
	}

	void Adam::optimizeOnCPU(std::unique_ptr<layer::BaseLayer>& refLayer)
	{
		auto pLayer = refLayer.get();
		u32 order = mLayerOrderMapOnCPU[pLayer];
		

		std::vector<DataArray>& momentumLst = mMomentumOnCPU[order];
		std::vector<DataArray>& velocityLst = mVelocityOnCPU[order];

		mIteration++;

		f32 effectiveLr = mLearningRate * std::sqrtf(1.0f - std::powf(mBeta1, mIteration)) / (1.0f - std::powf(mBeta0, mIteration));
		for (u32 id = 0; id < momentumLst.size(); id++)
		{
			auto& m = momentumLst[id];
			auto& v = velocityLst[id];

			auto& param = getLayerParamOnCPU(refLayer)[id];
			auto& dParam = getLayerDParamOnCPU(refLayer)[id];

			for (u32 i = 0, f = param.size; i < f; i++)
			{
				auto dP = dParam[i];
				f32 tmpM = m[i] += (1 - mBeta0) * (dP - m[i]);
				f32 tmpV = v[i] += (1 - mBeta1) * (dP * dP - v[i]);

				param[i] -= effectiveLr * tmpM / (std::sqrtf(tmpV) + 1e-7);
			}
		}
	}
}