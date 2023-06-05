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
			mMaskOnGPU.size =  mBatchSize * mDataSize;
			MALLOC_ON_GPU(mMaskOnGPU);
			INITIALIZE_GPU_DATA_1(mMaskOnGPU);

			//åvéZåãâ Çäiî[Ç∑ÇÈÇΩÇﬂÇÃÉÅÉÇÉäämï€
			mForwardResultOnGPU.size = mBackwardResultOnGPU.size = mBatchSize * mDataSize;
			MALLOC_ON_GPU(mForwardResultOnGPU);
			MALLOC_ON_GPU(mBackwardResultOnGPU);
			INITIALIZE_GPU_DATA_0(mForwardResultOnGPU);
			INITIALIZE_GPU_DATA_0(mBackwardResultOnGPU);
		}

		void ReLU::forwardOnGPU()
		{
			dim3 block(16, 16);
			dim3 grid(
				(mDataSize + block.x - 1) / block.x,
				(mBatchSize + block.y - 1) / block.y);

			ReLUForward << <grid, block >> > (
				mForwardResultOnGPU.address,
				mInputDataOnGPU->address,
				mMaskOnGPU.address,
				mDataSize,
				mDataSize,
				mBatchSize); 
#if _DEBUG
			CHECK(cudaDeviceSynchronize());
#endif
		}

		void ReLU::backwardOnGPU()
		{
			dim3 block(16, 16);
			dim3 grid(
				(mDataSize + block.x - 1) / block.x,
				(mBatchSize + block.y - 1) / block.y);

			ReLUBackward << <grid, block >> > (
				mBackwardResultOnGPU.address,
				mDInputDataOnGPU->address,
				mMaskOnGPU.address,
				mDataSize,
				mDataSize,
				mBatchSize); 
		}

		void ReLU::terminateOnGPU()
		{

		}
	}
}