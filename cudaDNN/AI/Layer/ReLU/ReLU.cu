#include <random>
#include <cuda_runtime.h>
#include <cassert>

#include "ReLU.h"
#include "../../../commonOnlyGPU.cuh"

namespace Aoba {
	namespace layer
	{
		namespace
		{
			__global__ void ReLUForward(
				f32* y, f32* x,
				f32* mask, u32 outputSize, u32 inputSize, u32 batchSize)
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
				//printf("%lf\n",y[id]);
			}



			__global__ void ReLUBackward(f32* y, f32* x,
				f32* mask, u32 outputSize, u32 inputSize, u32 batchSize)
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


		void ReLU::mallocOnGPU()
		{
			mMaskOnGPU.size =  mBatchSize * mInputSize;
			CHECK(cudaMalloc((void**)(&mMaskOnGPU.address), mMaskOnGPU.size * sizeof(f32)));
			{
				f32 * mask = new f32[mMaskOnGPU.size];
				for (u32 i = 0; i < mMaskOnGPU.size; i++)
				{
					mask[i] = 1.0f;
				}
				CHECK(cudaMemcpy(mMaskOnGPU.address, mask, mMaskOnGPU.size * sizeof(f32), cudaMemcpyHostToDevice));

			}

			//ŒvŽZŒ‹‰Ê‚ðŠi”[‚·‚é‚½‚ß‚Ìƒƒ‚ƒŠŠm•Û
			mForwardResultOnGPU.size = mBatchSize * mOutputSize;
			mBackwardResultOnGPU.size = mBatchSize * mInputSize;
			CHECK(cudaMalloc((void**)(&(mForwardResultOnGPU.address)),
				mForwardResultOnGPU.size * sizeof(f32)));
			CHECK(cudaMalloc((void**)(&(mBackwardResultOnGPU.address)),
				mBackwardResultOnGPU.size * sizeof(f32)));
			{
				f32* tmp = new f32[mForwardResultOnGPU.size];
				for (u32 idx = 0; idx < mForwardResultOnGPU.size; idx++)
				{
					tmp[idx] = 0.0f;
				}
				CHECK(cudaMemcpy(mForwardResultOnGPU.address, tmp,
					mForwardResultOnGPU.size * sizeof(f32), cudaMemcpyHostToDevice));
				delete[] tmp;


				tmp = new f32[mBackwardResultOnGPU.size];
				for (u32 idx = 0; idx < mBackwardResultOnGPU.size; idx++)
				{
					tmp[idx] = 0.0f;
				}
				CHECK(cudaMemcpy(mBackwardResultOnGPU.address, tmp,
					mBackwardResultOnGPU.size * sizeof(f32), cudaMemcpyHostToDevice));
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
				mMaskOnGPU.address,
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

			ReLUBackward << <grid, block >> > (
				mBackwardResultOnGPU.address,
				mDInputDataOnGPU->address,
				mMaskOnGPU.address,
				mOutputSize,
				mInputSize,
				mBatchSize);
		}

		void ReLU::terminateOnGPU()
		{

		}
	}
}