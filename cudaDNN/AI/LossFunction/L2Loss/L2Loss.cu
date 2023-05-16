#include "L2Loss.h"
#include "../../../commonOnlyGPU.cuh"

namespace Aoba
{
	namespace
	{
		__global__ void calcLoss(
			f32* forwardResult,
			f32* correctData,
			f32* dInput,
			f32* loss,
			u32 batchSize,
			u32 dataSize)
		{
			u32 batchID = blockIdx.x * blockDim.x + threadIdx.x;

			if (batchID >= batchSize)
			{
				return;
			}

			u32 offset = batchID * dataSize;
			

			f32 tmpLoss = 0.0f;
			for (u32 i = 0; i < dataSize; i++)
			{
				f32 sub = forwardResult[offset + i] - correctData[offset + i];
				tmpLoss += (0.5f * sub * sub);
				dInput[offset + i] = sub / batchSize;
			}

			loss[batchID] = tmpLoss;
		}
	}

	namespace lossFunction
	{
		void L2Loss::mallocOnGPU()
		{
			mDInputDataOnGPU.size = mBatchSize * mTrainingDataShape.getDataSize();
			CHECK(cudaMalloc((void**)(&(mDInputDataOnGPU.address)), mDInputDataOnGPU.size * sizeof(f32)));

			mLossTblOnGPU.size = mBatchSize;
			CHECK(cudaMalloc((void**)(&mLossTblOnGPU.address), mLossTblOnGPU.size * sizeof(f32)));
		}

		f32 L2Loss::calcLossAndDInputOnGPU()
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
			calcLoss << <grid, block >> > (
				mForwardResultOnGPU->address,
				mCorrectDataOnGPU->address,
				mDInputDataOnGPU.address,
				mLossTblOnGPU.address,
				mBatchSize,
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