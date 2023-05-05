#include "CrossEntropyWithSM.h"
#include "../../../commonGPU.cuh"

namespace Aoba
{
	namespace
	{
		__global__ void calcLoss(f32* forwardResult, f32* correctData, f32* dInput, f32* loss, u32 batchSize, u32 width, u32 dataSize)
		{
			u32 batchID = blockIdx.x * blockDim.x + threadIdx.x;

			if (batchID >= batchSize)
			{
				return;
			}

			u32 offset = batchID * dataSize;
			u32 correct = static_cast<u32>(correctData[batchID]);

			f32 max = forwardResult[offset + 0];
			u32 maxIndex = 0;
			f32 sum = 0.0f;
			for (u32 i = 0; i < width; i++)
			{
				f32 cand = forwardResult[offset + i];
				if (max < cand)
				{
					max = cand;
					maxIndex = i;
				}
			}

			for (u32 i = 0; i < width; i++)
			{
				sum += exp(forwardResult[offset + i] - max);
			}

			for (u32 i = 0; i < width; i++)
			{
				dInput[offset + i] = ((exp(forwardResult[offset + i] - max) / sum) - (correct == i ? 1 : 0)) / batchSize;
			}
			
			loss[batchID] = -log(exp(forwardResult[offset + correct] - max) / sum + 1e-7);
		}
	}

	namespace lossFunction
	{
		void CrossEntropyWithSM::mallocOnGPU()
		{
			mDInputDataOnGPU.size =mBatchSize * mTrainingDataShape.width;
			CHECK(cudaMalloc((void**)(&(mDInputDataOnGPU.address)), mDInputDataOnGPU.size * sizeof(f32)));

			mLossTblOnGPU.size = mBatchSize;
			CHECK(cudaMalloc((void**)(&mLossTblOnGPU.address), mLossTblOnGPU.size * sizeof(f32)));
		}

		f32 CrossEntropyWithSM::calcLossAndDInputOnGPU()
		{
			dim3 block(16, 1);
			dim3 grid(
				(mBatchSize + block.x - 1) / block.x);

			//エラーが出る（今後のために残しておく）
			/*f32* lossTblOnGPU = nullptr;
			CHECK(cudaMalloc((void**)(&lossTblOnGPU), mDataShape.batchSize * sizeof(f32)));*/
#if _DEBUG
			assert(mBatchSize != 0);
#endif
			calcLoss<<<grid, block>>>(
				mForwardResultOnGPU->address,
				mCorrectDataOnGPU->address,
				mDInputDataOnGPU.address, 
				mLossTblOnGPU.address, 
				mBatchSize,
				mTrainingDataShape.width,
				mForwardResultOnGPU->size / mBatchSize);
#if _DEBUG
			CHECK(cudaDeviceSynchronize());
#endif
			f32 loss = 0;
			std::vector<f32> lossOnCPU(mLossTblOnGPU.size);
			CHECK(cudaMemcpy(lossOnCPU.data(), mLossTblOnGPU.address, mLossTblOnGPU.size * sizeof(f32), cudaMemcpyDeviceToHost));

			for (u32 i = 0; i < mLossTblOnGPU.size; i++)
			{
				loss += lossOnCPU[i];
			}


			return loss / mBatchSize;
		}
	}
}