#include "CrossEntropyWithSM.h"
#include "../../../commonGPU.cuh"

namespace Aoba
{
	namespace
	{
		__global__ void calc(f32* forwardResult, f32* label, f32* dInput, f32* loss, u32 batchSize, u32 width, u32 dataSize)
		{
			u32 batchID = blockIdx.x * blockDim.x + threadIdx.x;

			if (batchID >= batchSize)
			{
				return;
			}

			u32 offset = batchID * dataSize;
			f32 correct = label[batchID];

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

			loss[batchID] = log(exp(forwardResult[offset + maxIndex]) / sum + 1e-7);
		}
	}

	namespace lossFunction
	{
		void CrossEntropyWithSM::initializeOnGPU()
		{
			mDInputData.size = mDataShape.batchSize * mDataShape.channel * mDataShape.height * mDataShape.width;
			CHECK(cudaMalloc((void**)(&(mDInputData.address)), mDInputData.size * sizeof(f32)));
		}

		f32 CrossEntropyWithSM::calcLossAndDInputOnGPU()
		{
			dim3 block(16, 1);
			dim3 grid(
				(mDataShape.batchSize + block.x - 1) / block.x);

			std::vector<f32> lossTbl(mDataShape.batchSize);
			f32* lossTblOnGPU = nullptr;
			CHECK(cudaMalloc((void**)(&(lossTblOnGPU)), mDataShape.batchSize * sizeof(f32)));
			calc<<<grid, block>>>(mForwardResult->address, mLabelData->address, mDInputData.address, lossTblOnGPU, mDataShape.batchSize, mDataShape.width, mDataShape.channel * mDataShape.height * mDataShape.width);
			CHECK(cudaMemcpy(lossTbl.data(), lossTblOnGPU, mDataShape.batchSize * sizeof(f32), cudaMemcpyDeviceToHost));
			f32 loss = 0;
			f32* pDataOnCPU = new f32[mForwardResult->size];
			f32* tmp = new f32[mForwardResult->size];
			CHECK(cudaMemcpy(pDataOnCPU, mForwardResult->address, mForwardResult->size * sizeof(f32), cudaMemcpyDeviceToHost));

			//‘¹Ž¸ŒvŽZ
			f32* labels = mLabelData->address;
			for (u32 id = 0; id < mDataShape.batchSize; id++)
			{
				u32 offset = id * (mDataShape.channel * mDataShape.height * mDataShape.width);
				u32 label = static_cast<u32>(labels[id]);

				f32 max = -1000.0f;
				f32 sum = 0.0f;
				for (u32 i = 0; i < mDataShape.width; i++)
				{
					f32 cand = pDataOnCPU[offset + i];
					if (max < cand)
					{
						max = cand;
					}
				}

				for (u32 i = 0; i < mDataShape.width; i++)
				{
					sum += std::expf(pDataOnCPU[offset + i] - max);
				}

				for (u32 i = 0; i < mDataShape.width; i++)
				{
					tmp[offset + i] = ((std::expf(pDataOnCPU[offset + i] - max) / sum) - (label == i ? 1 : 0)) / mDataShape.batchSize;
				}

				loss += -std::logf(std::expf(pDataOnCPU[offset + label] - max) / sum + 1e-7);
			}

			//‹t“`”À‚ÌŒvŽZ
#if _DEBUG
			std::vector<f32> t(mDInputData.size);
			for (int i = 0; i < t.size(); i++)t[i] = tmp[i];
#endif

			CHECK(cudaMemcpy(mDInputData.address, tmp, mDInputData.size * sizeof(f32), cudaMemcpyHostToDevice));



			delete[] tmp;
			delete[] pDataOnCPU;
			return loss / mDataShape.batchSize;
		}
	}
}