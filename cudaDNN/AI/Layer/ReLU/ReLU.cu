#include <random>
#include <cuda_runtime.h>
#include <cassert>

#include "ReLU.h"
#include "../../../commonGPU.cuh"

namespace Aoba {
	namespace layer
	{
		namespace
		{
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
			mMask.size =  mBatchSize * mInputSize;
			CHECK(cudaMalloc((void**)(&mMask.address), mMask.size * sizeof(f32)));
			{
				f32 * mask = new flowDataType[mMask.size];
				for (u32 i = 0; i < mMask.size; i++)
				{
					mask[i] = 1.0f;
				}
				CHECK(cudaMemcpy(mMask.address, mask, mMask.size * sizeof(flowDataType), cudaMemcpyHostToDevice));

			}

			//�v�Z���ʂ��i�[���邽�߂̃������m��
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

			std::vector<f32> forwardResultOnGPU(mForwardResultOnGPU.size);
			std::vector<f32> inputDataOnGPU(mInputDataOnGPU->size);
			std::vector<f32> mask(mMask.size);
			CHECK(cudaMemcpy(forwardResultOnGPU.data(), mForwardResultOnGPU.address, forwardResultOnGPU.size(), cudaMemcpyDeviceToHost));
			CHECK(cudaMemcpy(inputDataOnGPU.data(), (*mInputDataOnGPU).address, inputDataOnGPU.size(), cudaMemcpyDeviceToHost));
			CHECK(cudaMemcpy(mask.data(), mMask.address, mask.size(), cudaMemcpyDeviceToHost));
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
				mMask.address,
				mOutputSize,
				mInputSize,
				mBatchSize);
#if _DEBUG
			CHECK(cudaDeviceSynchronize());

			std::vector<f32> backwardResultOnGPU(mForwardResultOnGPU.size);
			std::vector<f32> dInputDataOnGPU(mInputDataOnGPU->size);
			std::vector<f32> mask(mMask.size);
			CHECK(cudaMemcpy(backwardResultOnGPU.data(), mBackwardResultOnGPU.address, backwardResultOnGPU.size(), cudaMemcpyDeviceToHost));
			CHECK(cudaMemcpy(dInputDataOnGPU.data(), (*mDInputDataOnGPU).address, dInputDataOnGPU.size(), cudaMemcpyDeviceToHost));
			CHECK(cudaMemcpy(mask.data(), mMask.address, mask.size(), cudaMemcpyDeviceToHost));
			CHECK(cudaDeviceSynchronize());
#endif
		}

		void ReLU::terminateOnGPU()
		{

		}
	}
}