#include <random>
#include <cuda_runtime.h>
#include <cassert>

//このマクロはCUDAファイルがコンパイルされる時に定義される。
//インテリセンスのエラーを一時的に抑制するためにこの定義を置いている。
#if !defined(__CUDACC__)
#define __CUDACC__
#endif

#include <device_functions.h>


#include "Convolution.h"
#include "../../../commonOnlyGPU.cuh"
#include "../../../common.h"


namespace Aoba {
	namespace layer
	{
		namespace
		{


			__global__ void ConvolutionForward(
				f32* y, f32* A,
				f32* x, f32* b, u32 outputSize, u32 inputSize, u32 batchSize)
			{
				u32 xid = blockIdx.x * blockDim.x + threadIdx.x;
				u32 yid = blockIdx.y * blockDim.y + threadIdx.y;
				if (xid >= outputSize || yid >= batchSize)
				{
					return;
				}
				u32 id = yid * outputSize + xid;

				f32 result = 0.0f;
				for (u32 i = 0; i < inputSize; i++)
				{
					//#if _DEBUG
					//					u32 tmp = xid * inputSize + i;
					//					if (tmp < 0 || tmp >= inputSize * outputSize)
					//					{
					//						printf("Affine A parameter : out of range : %d\n", tmp);
					//						printf("threadId x = %d  ,  y = %d\n", threadIdx.x, threadIdx.y);
					//						assert(0);
					//					}
					//					tmp = yid * inputSize + i;
					//					if (tmp < 0 || tmp >= inputSize * batchSize)
					//					{
					//						printf("Affine x parameter : out of range : %d", tmp);
					//						assert(0);
					//					}
					//#endif
					result += A[xid * inputSize + i] * x[yid * inputSize + i];
				}
				//#if _DEBUG
				//				if (!(id >= 0 && id < batchSize * outputSize))
				//				{
				//					printf("Affine y parameter : out of range : %d", id);
				//					assert(0);
				//				}
				//#endif
				y[id] = result + b[xid];
			}


			__global__ void FilterBackward(f32* dA, f32* dout, f32* input, u32 outputSize, u32 inputSize, u32 batchSize)
			{
				u32 xid = blockIdx.x * blockDim.x + threadIdx.x;
				u32 yid = blockIdx.y * blockDim.y + threadIdx.y;
				if (xid >= inputSize || yid >= outputSize)
				{
					return;
				}

				u32 id = yid * inputSize + xid;

				f32 result = 0.0f;
				for (u32 N = 0; N < batchSize; N++)
				{
#if _DEBUG
					if (N * inputSize + xid >= batchSize * inputSize)
					{
						assert(0);
					}
					if (N * outputSize + yid >= batchSize * outputSize)
					{
						assert(0);
					}
#endif
					result += dout[N * outputSize + yid] * input[N * inputSize + xid];
				}

				dA[id] = result;
				//printf("dA[%d]=%lf\n", id,result);
			}

			__global__ void biasBackward(f32* dBias, f32* dout, u32 outputSize, u32 batchSize)
			{
				u32 id = blockIdx.x * blockDim.x + threadIdx.x;
				if (id >= outputSize)
				{
					return;
				}
				f32 result = 0.0f;
				for (u32 N = 0; N < batchSize; N++)
				{
#if _DEBUG
					if ((N * outputSize + id) >= batchSize * outputSize)
					{
						assert(0);
					}
#endif
					result += dout[N * outputSize + id];
				}
#if _DEBUG
				if (id >= outputSize)
				{
					assert(0);
				}
#endif
				dBias[id] = result;
				//printf("%lf\n", result);
			}

			__global__ void doutBackward(f32* dOut, f32* A, f32* dIn, u32 outputSize, u32 inputSize, u32 batchSize)
			{
				u32 xid = blockIdx.x * blockDim.x + threadIdx.x;//input
				u32 yid = blockIdx.y * blockDim.y + threadIdx.y;//batch

				if (xid >= inputSize || yid >= batchSize)
				{
					return;
				}

				f32 result = 0.0f;
				for (u32 i = 0; i < outputSize; i++)
				{
#if _DEBUG
					if (i * inputSize + xid >= outputSize * inputSize)
					{
						assert(0);
					}
					if (yid * outputSize + i >= batchSize * outputSize)
					{
						assert(0);
					}
#endif
					result += A[i * inputSize + xid] * dIn[yid * outputSize + i];
				}
				dOut[yid * inputSize + xid] = result;
				//printf("dOut[%d * %d + %d] = %lf\n",yid, inputSize, xid, dOut[yid * inputSize + xid]);
			}
		}
		void Convolution::mallocOnGPU()
		{
			mParametersPtrOnGPU.resize(2);
			mDParametersPtrOnGPU.resize(2);

			//Affineパラメータ
			DataArray& convParam = mParametersPtrOnGPU[0];
			DataArray& convDParam = mDParametersPtrOnGPU[0];

			convParam.size = convDParam.size = mFilterNum * mIcFhFw;

			MALLOC_ON_GPU(convParam);
			MALLOC_ON_GPU(convDParam);

			INITIALIZE_GPU_DATA_NORMAL(convParam, 1, mConvolutionParamWeight);
			INITIALIZE_GPU_DATA_0(convDParam);


			//Biasパラメータ
			DataArray& biasParam = mParametersPtrOnGPU[1];
			DataArray& biasDParam = mDParametersPtrOnGPU[1];

			biasParam.size = biasDParam.size = mOutputDataShape.channel;

			MALLOC_ON_GPU(biasParam);
			MALLOC_ON_GPU(biasDParam);

			INITIALIZE_GPU_DATA_0(biasParam);
			INITIALIZE_GPU_DATA_0(biasDParam);


			//計算結果を格納するためのメモリ確保
			mForwardResultOnGPU.setSizeAs4D(mBatchSize, mOc, mOh, mOw);
			mReshapedInputDataOnGPU.setSizeAs3D(mBatchSize, mOhOw, mIcFhFw);
			mBackwardResultOnGPU.setSizeAs4D(mBatchSize, mIc, mIh, mIw);



			MALLOC_ON_GPU(mForwardResultOnGPU);
			MALLOC_ON_GPU(mReshapedInputDataOnGPU);
			MALLOC_ON_GPU(mBackwardResultOnGPU);

			INITIALIZE_GPU_DATA_0(mForwardResultOnGPU);
			INITIALIZE_GPU_DATA_0(mReshapedInputDataOnGPU);
			INITIALIZE_GPU_DATA_0(mBackwardResultOnGPU);
		}

		void Convolution::forwardOnGPU()
		{
			////将来的にgroupSharedを利用したほうが早いと確定したらこの部分の後者で置き換える予定。
			////			{
			////				std::chrono::system_clock::time_point time = std::chrono::system_clock::now();
			////#if 0
			////				dim3 block(16, 16);
			////				dim3 grid(
			////					(mOutputSize + block.x - 1) / block.x,
			////					(mBatchSize + block.y - 1) / block.y);
			////
			////
			////				AffineForward << <grid, block >> > (
			////					mForwardResultOnGPU.address,
			////					mParametersPtrOnGPU[0].address,
			////					mInputDataOnGPU->address,
			////					mParametersPtrOnGPU[1].address,
			////					mOutputSize,
			////					mInputSize,
			////					mBatchSize);
			////
			////#if _DEBUG
			////				CHECK(cudaDeviceSynchronize());
			////#endif
			////
			////#else
			////				u32 sharedMemorySize = 48000 / sizeof(f32);
			////
			////
			////				const u32 BlockSize = std::min(static_cast<u32>(1 << 5), sharedMemorySize / (2 * mInputSize));
			////				if (BlockSize < 1)
			////				{
			////					std::cout << "BlockSize is less than 1\n";
			////					assert(0);
			////				}
			////				dim3 block(BlockSize, BlockSize);
			////				dim3 grid((mOutputSize + BlockSize - 1) / BlockSize, (mBatchSize + BlockSize - 1) / BlockSize);
			////
			////				AffineForwardWithSM << <grid, block, 2 * mInputSize * BlockSize * sizeof(f32) >> > (
			////					mForwardResultOnGPU.address,
			////					mParametersPtrOnGPU[0].address,
			////					mInputDataOnGPU->address,
			////					mParametersPtrOnGPU[1].address,
			////					mOutputSize,
			////					mInputSize,
			////					mBatchSize);
			////
			////#if _DEBUG
			////				CHECK(cudaDeviceSynchronize());
			////#endif
			////
			////#endif
			////				auto time2 = static_cast<f32>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - time).count() / 1000.0f);
			////				//std::cout << time2 << std::endl;
			////				return;
			////		}


			//std::chrono::system_clock::time_point time;


			//if (mWhich == 0)
			//{
			//	if (mNowComparing)
			//	{
			//		time = std::chrono::system_clock::now();
			//	}
			//	dim3 block(32, 32);
			//	dim3 grid(
			//		(mOutputSize + block.x - 1) / block.x,
			//		(mBatchSize + block.y - 1) / block.y);


			//	AffineForward << <grid, block >> > (
			//		mForwardResultOnGPU.address,
			//		mParametersPtrOnGPU[0].address,
			//		mInputDataOnGPU->address,
			//		mParametersPtrOnGPU[1].address,
			//		mOutputSize,
			//		mInputSize,
			//		mBatchSize);

			//	if (mNowComparing)
			//	{
			//		CHECK(cudaDeviceSynchronize());
			//		auto elapsedTime = static_cast<f32>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - time).count() / 1000.0f);

			//		mFunc0AveTime = (mFunc0AveTime * mFunc0CallCnt + elapsedTime) / (mFunc0CallCnt + 1);
			//		mFunc0CallCnt++;

			//		if (mFunc0CallCnt >= CaptureTimes)
			//		{
			//			mWhich = 1;
			//		}
			//	}
			//}
			//else if (mWhich = 1)
			//{
			//	if (mNowComparing)
			//	{
			//		time = std::chrono::system_clock::now();
			//	}

			//	u32 sharedMemorySize = 48000 / sizeof(f32);


			//	const u32 BlockSize = std::min(static_cast<u32>(1 << 5), sharedMemorySize / (2 * mInputSize));
			//	if (BlockSize < 1)
			//	{
			//		std::cout << "BlockSize is less than 1\n";
			//		assert(0);
			//	}
			//	dim3 block(BlockSize, BlockSize);
			//	dim3 grid((mOutputSize + BlockSize - 1) / BlockSize, (mBatchSize + BlockSize - 1) / BlockSize);

			//	AffineForwardWithSM << <grid, block, 2 * mInputSize * BlockSize * sizeof(f32) >> > (
			//		mForwardResultOnGPU.address,
			//		mParametersPtrOnGPU[0].address,
			//		mInputDataOnGPU->address,
			//		mParametersPtrOnGPU[1].address,
			//		mOutputSize,
			//		mInputSize,
			//		mBatchSize);

			//	if (mNowComparing)
			//	{
			//		CHECK(cudaDeviceSynchronize());
			//		auto elapsedTime = static_cast<f32>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - time).count() / 1000.0f);

			//		mFunc1AveTime = (mFunc1AveTime * mFunc1CallCnt + elapsedTime) / (mFunc1CallCnt + 1);
			//		mFunc1CallCnt++;

			//		if (mFunc1CallCnt >= CaptureTimes)
			//		{
			//			if (mFunc1AveTime > mFunc0AveTime)
			//			{
			//				mWhich = 0;
			//			}
			//			else
			//			{
			//				mWhich = 1;
			//			}
			//			mNowComparing = false;
			//		}
			//	}

			//}
			//else
			//{
			//	assert(0);
			//}
		}

		void Convolution::backwardOnGPU()
		{
//			//doutの逆伝搬
//			{
//				dim3 block(16, 16);
//				dim3 grid(
//					(mInputSize + block.x - 1) / block.x,
//					(mBatchSize + block.y - 1) / block.y);
//				doutBackward << <grid, block >> > (
//					mBackwardResultOnGPU.address,
//					mParametersPtrOnGPU[0].address,
//					mDInputDataOnGPU->address,
//					mOutputSize,
//					mInputSize,
//					mBatchSize);
//#if _DEBUG
//				CHECK(cudaDeviceSynchronize());
//#endif
//			}
//
//			//Aの逆伝搬
//			{
//				dim3 block(16, 16);
//				dim3 grid(
//					(mInputSize + block.x - 1) / block.x,
//					(mOutputSize + block.y - 1) / block.y);
//
//				AffineBackward << <grid, block >> > (
//					pDParametersOnGPU[0].address,
//					mDInputDataOnGPU->address,
//					mInputDataOnGPU->address,
//					mOutputSize,
//					mInputSize,
//					mBatchSize);
//
//#if _DEBUG
//				CHECK(cudaDeviceSynchronize());
//#endif
//			}
//
//			//Biasの逆伝搬
//			{
//				dim3 block(16);
//				dim3 grid((mOutputSize + block.x - 1) / block.x);
//
//				biasBackward << <grid, block >> > (
//					pDParametersOnGPU[1].address,
//					mDInputDataOnGPU->address,
//					mOutputSize,
//					mBatchSize);
//
//#if _DEBUG
//				CHECK(cudaDeviceSynchronize());
//#endif
//			}
		}

		void Convolution::terminateOnGPU()
		{

		}

	}
}