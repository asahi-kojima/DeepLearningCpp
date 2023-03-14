#include <random>
#include <cuda_runtime.h>
#include <cassert>

#include "ReLU.h"
#include "../../commonGPU.cuh"

namespace miduho {
	namespace layer
	{
		namespace
		{
			using flowDataType = BaseLayer::flowDataType;


			__global__ void ReLUForward(
				flowDataType* y, flowDataType* x,
				flowDataType* mask, u32 outputSize, u32 inputSize, u32 batchSize)
			{
				u32 xid = blockIdx.x * blockDim.x + threadIdx.x;
				u32 yid = blockIdx.y * blockDim.y + threadIdx.y;
				if (xid >= outputSize || yid >= batchSize)
				{
					return;
				}

				u32 id = yid * outputSize + xid;
#if _DEBUG
				if (id >= outputSize * batchSize)
				{
					printf("ReLU  : out of range : %d", id);
					assert(0);
				}
#endif
				f32 input = x[id];
				if (input < 0)
				{
					mask[id] = 0;
					y[id] = 0;
				}
				else
				{
					mask[id] = 1;
					y[id] = input;
				}
			}



			__global__ void ReLUBackward(flowDataType* y, flowDataType* x,
				flowDataType* mask, u32 outputSize, u32 inputSize, u32 batchSize)
			{
				u32 xid = blockIdx.x * blockDim.x + threadIdx.x;
				u32 yid = blockIdx.y * blockDim.y + threadIdx.y;
				if (xid >= inputSize || yid >= batchSize)
				{
					return;
				}

				u32 id = yid * inputSize + xid;
#if _DEBUG
				if (id >= inputSize * batchSize)
				{
					printf("ReLU  : out of range : %d", id);
					assert(0);
				}
#endif
				y[id] = x[id] * mask[id];
			}
		}


		void ReLU::initializeOnGPU()
		{
			mMask.size = mDefaultMask.size = mInputSize;
			CHECK(cudaMalloc((void**)(&mMask.address), mMask.size * sizeof(f32)));
			{
				mDefaultMask.address = new flowDataType[mMask.size];
				for (u32 i = 0; i < mMask.size; i++)
				{
					mDefaultMask.address[i] = 1.0f;
				}
				CHECK(cudaMemcpy(mMask.address, mDefaultMask.address, mMask.size * sizeof(flowDataType), cudaMemcpyHostToDevice));

			}

			//ŒvŽZŒ‹‰Ê‚ðŠi”[‚·‚é‚½‚ß‚Ìƒƒ‚ƒŠŠm•Û
			mForwardResultOnGPU.size = mBatchSize * mOutputSize;
			mBackwardResultOnGPU.size = mBatchSize * mInputSize;
			CHECK(cudaMalloc((void**)(&(mForwardResultOnGPU.address)),
				mForwardResultOnGPU.size * sizeof(flowDataType)));
			CHECK(cudaMalloc((void**)(&(mBackwardResultOnGPU.address)),
				mBackwardResultOnGPU.size * sizeof(flowDataType)));
			{
				flowDataType* tmp = new flowDataType[mForwardResultOnGPU.size];
				for (u32 idx = 0; idx < mForwardResultOnGPU.size; idx++)
				{
					tmp[idx] = 0.0f;
				}
				CHECK(cudaMemcpy(mForwardResultOnGPU.address, tmp,
					mForwardResultOnGPU.size * sizeof(flowDataType), cudaMemcpyHostToDevice));
				delete[] tmp;


				tmp = new flowDataType[mBackwardResultOnGPU.size];
				for (u32 idx = 0; idx < mBackwardResultOnGPU.size; idx++)
				{
					tmp[idx] = 0.0f;
				}
				CHECK(cudaMemcpy(mBackwardResultOnGPU.address, tmp,
					mBackwardResultOnGPU.size * sizeof(flowDataType), cudaMemcpyHostToDevice));
				delete[] tmp;
			}
		}

		void ReLU::forwardOnGPU()
		{
			dim3 block(16, 16);
			dim3 grid(
				(mOutputSize + block.x - 1) / block.x,
				(mBatchSize + block.y - 1) / block.y);

			ReLUForward << <grid, block >> > (
				mForwardResultOnGPU.address,
				mInputDataOnGPU->address,
				mMask.address,
				mOutputSize,
				mInputSize,
				mBatchSize);
#if _DEBUG
			CHECK(cudaDeviceSynchronize());
#endif
		}

		void ReLU::backwardOnGPU()
		{
			dim3 block(16, 16);
			dim3 grid(
				(mInputSize + block.x - 1) / block.x,
				(mBatchSize + block.y - 1) / block.y);

			ReLUForward << <grid, block >> > (
				mBackwardResultOnGPU.address,
				mDInputDataOnGPU->address,
				mMask.address,
				mOutputSize,
				mInputSize,
				mBatchSize);
#if _DEBUG
			CHECK(cudaDeviceSynchronize());
#endif
		}

		void ReLU::terminateOnGPU()
		{

		}
	}
}