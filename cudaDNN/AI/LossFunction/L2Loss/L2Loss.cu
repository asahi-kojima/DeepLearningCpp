#include "L2Loss.h"

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
#if INDEX_DEBUG
			assert(mBatchSize != 0);
#endif
#if TIME_DEBUG
			{
				std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
				calcLoss << <grid, block >> > (
					mForwardResultOnGPU->address,
					mCorrectDataOnGPU->address,
					mDInputDataOnGPU.address,
					mLossTblOnGPU.address,
					mBatchSize,
					mForwardResultOnGPU->size / mBatchSize);
#if GPU_SYNC_DEBUG
				CHECK(cudaDeviceSynchronize());
#endif
#if TIME_DEBUG
				f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
				std::string name = "";
				(((name += __FUNCTION__) += " : ") ) += " : calcLoss";
				timers[name] = elapsedTime;
			}
#endif
			std::vector<f32> lossOnCPU(mLossTblOnGPU.size);
			CHECK(cudaMemcpy(lossOnCPU.data(), mLossTblOnGPU.address, mLossTblOnGPU.size * sizeof(f32), cudaMemcpyDeviceToHost));

			f32 loss = 0;
			for (u32 i = 0; i < mLossTblOnGPU.size; i++)
			{
				loss += lossOnCPU[i];
			}


			return loss / mBatchSize;
		}
	}
}