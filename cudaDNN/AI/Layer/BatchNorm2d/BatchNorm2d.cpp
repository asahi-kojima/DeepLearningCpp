//#include <iostream>
//#include <cassert>
//
//#include "../../../commonCPU.h"
//#include "BatchNorm2d.h"
//
//namespace Aoba::layer
//{
//	BatchNorm2d::BatchNorm2d()
//		: mBatchSize(0)
//		, mInputSize(0)
//		, mOutputSize(0)
//	{}
//
//	BatchNorm2d::~BatchNorm2d() {}
//
//	//////////////////////////////////////
//	//CPU ä÷êî
//	//////////////////////////////////////
//	void BatchNorm2d::initializeOnCPU()
//	{
//		pParametersOnCPU.resize(2);
//		pDParametersOnCPU.resize(2);
//
//
//		//Gamma
//		paramMemory& gammaParam = pParametersOnCPU[0];
//		paramMemory& gammaDParam = pDParametersOnCPU[0];
//
//		gammaParam.size = mOutputSize;
//		gammaDParam.size = mOutputSize;
//
//		gammaParam.address = new f32[gammaParam.size];
//		gammaDParam.address = new f32[gammaDParam.size];
//
//		for (u32 i = 0; i < gammaParam.size; i++)
//		{
//			gammaParam.address[i] = 1.0f;
//			gammaDParam.address[i] = 1.0f;
//		}
//
//		//Beta
//		paramMemory& betaParam = pParametersOnCPU[1];
//		paramMemory& betaDParam = pDParametersOnCPU[1];
//
//		betaParam.size = mOutputSize;
//		betaDParam.size = mOutputSize;
//
//		betaParam.address = new f32[betaParam.size];
//		betaDParam.address = new f32[betaDParam.size];
//
//		for (u32 i = 0; i < betaParam.size; i++)
//		{
//			betaParam.address[i] = 1.0f;
//			betaDParam.address[i] = 1.0f;
//		}
//
//
//
//		//ì`î¿óp
//		mForwardResultOnCPU.size = mBatchSize * mOutputSize;
//		mBackwardResultOnCPU.size = mBatchSize * mInputSize;
//
//		mForwardResultOnCPU.address = new f32[mForwardResultOnCPU.size];
//		mBackwardResultOnCPU.address = new f32[mBackwardResultOnCPU.size];
//
//		for (u32 i = 0; i < mForwardResultOnCPU.size; i++)
//		{
//			mForwardResultOnCPU.address[i] = 1.0f;
//			mBackwardResultOnCPU.address[i] = 1.0f;
//		}
//	}
//
//	void BatchNorm2d::forwardOnCPU()
//	{
//		for (u32 c = 0; c < mChannel; c++)
//		{
//			f32 mean = 0.0f;
//			for (u32 N = 0; N < mBatchSize; N++)
//			{
//				for (u32 o = 0; o < mOutputSize; o++)
//				{
//					for (u32 i = 0; i < mInputSize; i++)
//					{
//						u32 index = 0;
//						mean += 0;
//					}
//				}
//			}
//		}
//	}
//
//	void BatchNorm2d::backwardOnCPU()
//	{
//
//	}
//
//	void BatchNorm2d::terminateOnCPU()
//	{
//		delete[] mForwardResultOnCPU.address;
//		delete[] mBackwardResultOnCPU.address;
//	}
//}