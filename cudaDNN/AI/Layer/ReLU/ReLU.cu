#include <random>
#include <cuda_runtime.h>
#include <cassert>

#include "ReLU.h"

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
#if INDEX_DEBUG
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
#if INDEX_DEBUG
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
			mMaskOnGPU.size = mBatchSize * mDataSize;
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
#if TIME_DEBUG
			{
				std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
				ReLUForward << <grid, block >> > (
					mForwardResultOnGPU.address,
					mInputDataOnGPU->address,
					mMaskOnGPU.address,
					mDataSize,
					mDataSize,
					mBatchSize);
#if GPU_SYNC_DEBUG
				CHECK(cudaDeviceSynchronize());
#endif
#if TIME_DEBUG
				f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
				std::string name = "";
				(((name += __FUNCTION__) += " : ") += std::to_string(mInstanceID)) += " : ReLUForward";
				timers[name] = elapsedTime;
			}
#endif
		}

		void ReLU::backwardOnGPU()
		{
			dim3 block(16, 16);
			dim3 grid(
				(mDataSize + block.x - 1) / block.x,
				(mBatchSize + block.y - 1) / block.y);
#if TIME_DEBUG
			{
				std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
				ReLUBackward << <grid, block >> > (
					mBackwardResultOnGPU.address,
					mDInputDataOnGPU->address,
					mMaskOnGPU.address,
					mDataSize,
					mDataSize,
					mBatchSize);
#if GPU_SYNC_DEBUG
				CHECK(cudaDeviceSynchronize());
#endif
#if TIME_DEBUG
				f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
				std::string name = "";
				(((name += __FUNCTION__) += " : ") += std::to_string(mInstanceID)) += " : ReLUBackward";
				timers[name] = elapsedTime;
			}
#endif
		}

		void ReLU::terminateOnGPU()
		{

		}
	}
}