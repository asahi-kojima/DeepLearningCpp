#include "CrossEntropyWithSM.h"


namespace Aoba
{
	namespace lossFunction
	{
		void CrossEntropyWithSM::mallocOnCPU()
		{
			mDInputDataOnCPU.size = mBatchSize * mTrainingDataShape.channel * mTrainingDataShape.height * mTrainingDataShape.width;
			mDInputDataOnCPU.address = new f32[mDInputDataOnCPU.size];

			mLossTblOnCPU.size = mBatchSize;
			mLossTblOnCPU.address = new f32[mLossTblOnCPU.size];
		}

		f32 CrossEntropyWithSM::calcLossAndDInputOnCPU()
		{
			DataMemory& input = *mForwardResultOnCPU;
			DataMemory& correctData = *mCorrectDataOnCPU;

			u32 batchSize = mBatchSize;
			u32 dataSize = input.size / batchSize;

			f32 loss = 0.0f;

			for (u32 batchID = 0; batchID < batchSize; batchID++)
			{
				u32 offset = batchID * dataSize;
				u32 correct = static_cast<u32>(correctData.address[batchID]);


				f32 max = input.address[offset + 0];
				for (u32 i = 0; i < dataSize; i++)
				{
					f32 cand = input.address[offset + i];
					if (max < cand)
					{
						max = cand;
					}
				}


				f32 sum = 0;
				for (u32 i = 0; i < dataSize; i++)
				{
					sum += exp(input.address[offset + i] - max);
				}

				for (u32 i = 0; i < dataSize; i++)
				{
					mDInputDataOnCPU.address[offset + i] = ((exp(input.address[offset + i] - max) / sum) - (correct == i ? 1 : 0)) / batchSize;
				}

				loss += mLossTblOnCPU.address[batchID] = -log(exp(input.address[offset + correct] - max) / sum + 1e-7);;
			}

#if _DEBUG
			assert(mBatchSize != 0);
#endif
			return loss / mBatchSize;
		}
	}
}