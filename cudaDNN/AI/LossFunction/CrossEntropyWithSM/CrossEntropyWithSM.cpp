#include "CrossEntropyWithSM.h"


namespace Aoba
{
	namespace lossFunction
	{
		void CrossEntropyWithSM::mallocOnCPU()
		{
			mDInputDataOnCPU.setSizeAs4D(mBatchSize , mTrainingDataShape.channel , mTrainingDataShape.height , mTrainingDataShape.width);
			MALLOC_AND_INITIALIZE_0_ON_CPU(mDInputDataOnCPU);

			mLossTblOnCPU.size = mBatchSize;
			MALLOC_AND_INITIALIZE_0_ON_CPU(mLossTblOnCPU);
		}

		f32 CrossEntropyWithSM::calcLossAndDInputOnCPU()
		{
			DataArray& input = *mForwardResultOnCPU;
			DataArray& correctData = *mCorrectDataOnCPU;

			const u32 dataSize = input.size / mBatchSize;

			f32 loss = 0.0f;

			for (u32 batchID = 0; batchID < mBatchSize; batchID++)
			{
				u32 offset = batchID * dataSize;
				u32 correct = static_cast<u32>(correctData[batchID]);


				f32 max = input[offset + 0];
				for (u32 i = 0; i < dataSize; i++)
				{
					f32 cand = input[offset + i];
					if (max < cand)
					{
						max = cand;
					}
				}


				f32 sum = 0;
				for (u32 i = 0; i < dataSize; i++)
				{
					sum += exp(input[offset + i] - max);
				}

				for (u32 i = 0; i < dataSize; i++)
				{
					mDInputDataOnCPU[offset + i] = ((exp(input[offset + i] - max) / sum) - (correct == i ? 1 : 0));
				}

				loss += mLossTblOnCPU[batchID] = -log(exp(input[offset + correct] - max) / sum + 1e-7);
			}
			return loss / mBatchSize;
		}

		void CrossEntropyWithSM::terminateOnCPU()
		{
			delete[] mDInputDataOnCPU.address;
			delete[] mLossTblOnCPU.address;
		}
	}
}