#include "CrossEntropyWithSM.h"
#include "../../../commonGPU.cuh"

namespace Aoba
{
	namespace lossFunction
	{
		void CrossEntropyWithSM::initializeOnGPU()
		{
			mDInputData.size = mDataShape.batchSize * mDataShape.channel * mDataShape.height * mDataShape.width;
			CHECK(cudaMalloc((void**)(&(mDInputData.address)), mDInputData.size * sizeof(parameterType)));
		}

		f32 CrossEntropyWithSM::calcLossAndDInputOnGPU(constDataMemory& data, void* label)
		{
			f32 loss = 0;
			f32* pDataOnCPU = new f32[data.size];
			f32* tmp = new f32[data.size];
			CHECK(cudaMemcpy(pDataOnCPU, data.address, data.size * sizeof(f32), cudaMemcpyDeviceToHost));

			//‘¹Ž¸ŒvŽZ
			f32* labels = reinterpret_cast<f32*>(label);
			for (u32 id = 0; id < mDataShape.batchSize; id++)
			{
				u32 offset = id * (mDataShape.channel * mDataShape.height * mDataShape.width);
				u32 label = static_cast<u32>(labels[id]);
				
				f32 max =-1000.0f;
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

			CHECK(cudaMemcpy(mDInputData.address, tmp, mDInputData.size * sizeof(flowDataType), cudaMemcpyHostToDevice));



			delete[] tmp;
			delete[] pDataOnCPU;
			return loss / mDataShape.batchSize;
		}
	}
}