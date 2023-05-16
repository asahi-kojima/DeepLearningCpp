#include <iostream>
#include <cassert>

#include "../../../common.h"
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
		pParametersOnCPU.resize(2);
		pDParametersOnCPU.resize(2);


		//------------------------------------------------------------------
		//Gamma
		//------------------------------------------------------------------
		paramMemory& gammaParam = pParametersOnCPU[0];
		paramMemory& gammaDParam = pDParametersOnCPU[0];

		gammaParam.size = mDataShape.channel;
		gammaDParam.size = gammaParam.size;

		gammaParam.address = new f32[gammaParam.size];
		gammaDParam.address = new f32[gammaDParam.size];

		for (u32 i = 0; i < gammaParam.size; i++)
		{
			gammaParam.address[i] = 1.0f;
			gammaDParam.address[i] = 1.0f;
		}

		//------------------------------------------------------------------
		//Beta
		//------------------------------------------------------------------
		paramMemory& betaParam = pParametersOnCPU[1];
		paramMemory& betaDParam = pDParametersOnCPU[1];

		betaParam.size = mDataShape.channel;
		betaDParam.size = betaParam.size;

		betaParam.address = new f32[betaParam.size];
		betaDParam.address = new f32[betaDParam.size];

		for (u32 i = 0; i < betaParam.size; i++)
		{
			betaParam.address[i] = 0.0f;
			betaDParam.address[i] = 0.0f;
		}

		//------------------------------------------------------------------
		//Sigma
		//------------------------------------------------------------------
		mSigmaOnCPU.size = mDataShape.channel;
		mSigmaOnCPU.byteSize = mSigmaOnCPU.size * sizeof(f32);

		mSigmaOnCPU.address = new f32[mSigmaOnCPU.size];
		for (u32 i = 0; i < mSigmaOnCPU.size; i++)
		{
			mSigmaOnCPU.address[i] = 0.0f;
		}

		//------------------------------------------------------------------
		//ì`î¿óp
		//------------------------------------------------------------------
		mForwardResultOnCPU.size = mBatchSize * mDataShape.getDataSize();
		mBackwardResultOnCPU.size = mForwardResultOnCPU.size;
		mIntermediateResultOnCPU.size = mForwardResultOnCPU.size;

		mForwardResultOnCPU.address = new f32[mForwardResultOnCPU.size];
		mBackwardResultOnCPU.address = new f32[mBackwardResultOnCPU.size];
		mIntermediateResultOnCPU.address = new f32[mIntermediateResultOnCPU.size];


		for (u32 i = 0; i < mForwardResultOnCPU.size; i++)
		{
			mForwardResultOnCPU.address[i] = 0.0f;
			mBackwardResultOnCPU.address[i] = 0.0f;
			mIntermediateResultOnCPU.address[i] = 0.0f;
		}
	}

	void BatchNorm2d::forwardOnCPU()
	{
		f32 ep = 1e-7;
		u32 hXw = mDataShape.height * mDataShape.width;
		for (u32 c = 0; c < mDataShape.channel; c++)
		{
			f32 mean = 0.0f;
			f32 sigma = 0.0f;

			//------------------------------------------------------------------
			//ïΩãœÇåvéZ
			//------------------------------------------------------------------
			for (u32 N = 0; N < mBatchSize; N++)
			{
				for (u32 hw = 0; hw < hXw; hw++)
				{
					mean += mInputDataOnCPU->address[N * mDataShape.getDataSize() + c * hXw + hw];
				}
			}
			mean /= (mBatchSize * hXw);

			//------------------------------------------------------------------
			//ïŒç∑ÇåvéZ
			//------------------------------------------------------------------
			f32 sigma2 = 0.0f;
			for (u32 N = 0; N < mBatchSize; N++)
			{
				for (u32 hw = 0; hw < hXw; hw++)
				{
					f32 diff = mInputDataOnCPU->address[N * mDataShape.getDataSize() + c * hXw + hw] - mean;
					sigma2 += diff * diff;
				}
			}
			sigma2 /= (mBatchSize * hXw);
			sigma = std::sqrt(sigma2);

			//------------------------------------------------------------------
			//ïWèÄâª
			//------------------------------------------------------------------
			f32 gamma = pParametersOnCPU[0].address[c];
			f32 beta = pParametersOnCPU[1].address[c];
			for (u32 N = 0; N < mBatchSize; N++)
			{
				for (u32 hw = 0; hw < hXw; hw++)
				{
					u32 index = N * mDataShape.getDataSize() + c * hXw + hw;
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
			pDParametersOnCPU[0][c] = dGamma;
			pDParametersOnCPU[1][c] = dBeta;
			
			
			
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
					mBackwardResultOnCPU[index] = (pParametersOnCPU[0][c] / (mSigmaOnCPU[c] + 1e-7)) * (dout[index] - dMean - mIntermediateResultOnCPU[index] * diMean);
				}
			}
		}

	}

	void BatchNorm2d::terminateOnCPU()
	{
		delete[] mForwardResultOnCPU.address;
		delete[] mBackwardResultOnCPU.address;
	}
}