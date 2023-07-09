#include <iostream>
#include <cassert>
#include "BatchNorm2d.h"

namespace Aoba::layer
{
	BatchNorm2d::BatchNorm2d()
	{}

	BatchNorm2d::~BatchNorm2d() {}

	void BatchNorm2d::setupLayerInfo(u32 batchSize, DataShape& shape)
	{
		mBatchSize = batchSize;
		mDataShape = shape;
	}


	//////////////////////////////////////
	//CPU ä÷êî
	//////////////////////////////////////
	void BatchNorm2d::mallocOnCPU()
	{
		mParametersPtrOnCPU.resize(2);
		mDParametersPtrOnCPU.resize(2);


		//------------------------------------------------------------------
		//Gamma
		//------------------------------------------------------------------
		DataArray& gammaParam = mParametersPtrOnCPU[0];
		DataArray& gammaDParam = mDParametersPtrOnCPU[0];

		gammaParam.size = gammaDParam.size = mDataShape.channel;
		MALLOC_AND_INITIALIZE_1_ON_CPU(gammaParam);
		MALLOC_AND_INITIALIZE_1_ON_CPU(gammaDParam);

		//------------------------------------------------------------------
		//Beta
		//------------------------------------------------------------------
		DataArray& betaParam = mParametersPtrOnCPU[1];
		DataArray& betaDParam = mDParametersPtrOnCPU[1];

		betaParam.size = betaDParam.size = mDataShape.channel;
		MALLOC_AND_INITIALIZE_0_ON_CPU(betaParam);
		MALLOC_AND_INITIALIZE_0_ON_CPU(betaDParam);

		//------------------------------------------------------------------
		//Sigma
		//------------------------------------------------------------------
		mSigmaOnCPU.size = mDataShape.channel;
		mSigmaOnCPU.byteSize = mSigmaOnCPU.size * sizeof(f32);
		MALLOC_AND_INITIALIZE_0_ON_CPU(mSigmaOnCPU);

		//------------------------------------------------------------------
		//ì`î¿óp
		//------------------------------------------------------------------
		mForwardResultOnCPU.setSizeAs4D(mBatchSize, mDataShape);
		mBackwardResultOnCPU.setSizeAs4D(mBatchSize, mDataShape);
		mIntermediateResultOnCPU.setSizeAs4D(mBatchSize, mDataShape);
		MALLOC_AND_INITIALIZE_0_ON_CPU(mForwardResultOnCPU);
		MALLOC_AND_INITIALIZE_0_ON_CPU(mBackwardResultOnCPU);
		MALLOC_AND_INITIALIZE_0_ON_CPU(mIntermediateResultOnCPU);
	}

	void BatchNorm2d::forwardOnCPU()
	{
		f32 ep = 1e-7;
		u32 hXw = mDataShape.height * mDataShape.width;
		for (u32 c = 0; c < mDataShape.channel; c++)
		{
			f32 mean = 0.0f;
			f32 sqMean = 0.0f;
			f32 sigma = 0.0f;

			const u32 dataSize = mDataShape.getDataSize();
			//------------------------------------------------------------------
			//ïΩãœÇåvéZ
			//------------------------------------------------------------------
			for (u32 N = 0; N < mBatchSize; N++)
			{
				for (u32 hw = 0; hw < hXw; hw++)
				{
					f32 value = mInputDataOnCPU->address[N * dataSize + c * hXw + hw];
					mean += value;
					sqMean += value * value;
				}
			}
			mean /= (mBatchSize * hXw);
			sqMean /= (mBatchSize * hXw);

			//------------------------------------------------------------------
			//ïŒç∑ÇåvéZ
			//------------------------------------------------------------------
			/*f32 sigma2 = 0.0f;
			for (u32 N = 0; N < mBatchSize; N++)
			{
				for (u32 hw = 0; hw < hXw; hw++)
				{
					f32 diff = mInputDataOnCPU->address[N * mDataShape.getDataSize() + c * hXw + hw] - mean;
					sigma2 += diff * diff;
				}
			}
			sigma2 /= (mBatchSize * hXw);
			sigma = std::sqrt(sigma2);*/

			sigma = std::sqrt(sqMean - mean * mean);

			//------------------------------------------------------------------
			//ïWèÄâª
			//------------------------------------------------------------------
			f32 gamma = mParametersPtrOnCPU[0].address[c];
			f32 beta = mParametersPtrOnCPU[1].address[c];
			for (u32 N = 0; N < mBatchSize; N++)
			{
				for (u32 hw = 0; hw < hXw; hw++)
				{
					u32 index = N * dataSize + c * hXw + hw;
					f32 normalizeResult = (mInputDataOnCPU->address[index] - mean) / sigma;
					mIntermediateResultOnCPU.address[index] = normalizeResult;
					mForwardResultOnCPU.address[index] = gamma * normalizeResult + beta;
				}
			}

			mSigmaOnCPU.address[c] = sigma;
		}
	}

	void BatchNorm2d::backwardOnCPU()
	{
		u32 hXw = mDataShape.height * mDataShape.width;
		auto& dout = *mDInputDataOnCPU;
		for (u32 c = 0; c < mDataShape.channel; c++)
		{
			f32 dGamma = 0.0f;
			f32 dBeta = 0.0f;
			for (u32 N = 0; N < mBatchSize; N++)
			{
				for (u32 hw = 0; hw < hXw; hw++)
				{
					u32 index = N * mDataShape.getDataSize() + c * hXw + hw;
					f32 dO = dout[index];
					dGamma += dO * mIntermediateResultOnCPU[index];
					dBeta += dO;
				}
			}
			mDParametersPtrOnCPU[0][c] = dGamma;
			mDParametersPtrOnCPU[1][c] = dBeta;
			
			
			
			f32 dMean = 0.0f;
			f32 diMean = 0.0f;
			for (u32 N = 0; N < mBatchSize; N++)
			{
				for (u32 hw = 0; hw < hXw; hw++)
				{
					u32 index = N * mDataShape.getDataSize() + c * hXw + hw;
					dMean += dout[index];
					diMean += dout[index] * mIntermediateResultOnCPU[index];
				}
			}
			dMean /= (mBatchSize * hXw);
			diMean /= (mBatchSize * hXw);

			for (u32 N = 0; N < mBatchSize; N++)
			{
				for (u32 hw = 0; hw < hXw; hw++)
				{
					u32 index = N * mDataShape.getDataSize() + c * hXw + hw;
					mBackwardResultOnCPU[index] = (mParametersPtrOnCPU[0][c] / (mSigmaOnCPU[c] + 1e-7)) * (dout[index] - dMean - mIntermediateResultOnCPU[index] * diMean);
				}
			}
		}

	}

	void BatchNorm2d::terminateOnCPU()
	{
		for (u32 id = 0; id < mParametersPtrOnCPU.size(); id++)
		{
			delete[] mParametersPtrOnCPU[id].address;
			delete[] mDParametersPtrOnCPU[id].address;
		}

		delete[] mSigmaOnCPU.address;

		delete[] mForwardResultOnCPU.address;
		delete[] mBackwardResultOnCPU.address;
		delete[] mIntermediateResultOnCPU.address;
	}
}