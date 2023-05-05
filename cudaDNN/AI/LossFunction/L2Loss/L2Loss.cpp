#include "L2Loss.h"


namespace Aoba
{
	namespace lossFunction
	{
		void L2Loss::mallocOnCPU()
		{
			mDInputDataOnCPU.size = mBatchSize * mTrainingDataShape.getDataSize();
			mDInputDataOnCPU.address = new f32[mDInputDataOnCPU.size];

			mLossTblOnCPU.size = mBatchSize;
			mLossTblOnCPU.address = new f32[mLossTblOnCPU.size];
		}

		f32 L2Loss::calcLossAndDInputOnCPU()
		{
			DataMemory& input = *mForwardResultOnCPU;
			DataMemory& correctData = *mCorrectDataOnCPU;

			u32 dataSize = input.size / mBatchSize;

			f32 loss = 0.0f;

			if (mTrainingDataShape != mCorrectDataShape)
			{
				std::cout << "shape mismatch" << std::endl;
				assert(0);
			}

			for (u32 batchID = 0; batchID < mBatchSize; batchID++)
			{
				u32 offset = batchID * dataSize;

				f32 result = 0.0f;
				for (u32 index = 0; index < dataSize; index++)
				{
					f32 sub = input.address[offset + index] - correctData.address[offset + index];
					result += 0.5 * sub * sub;

					mDInputDataOnCPU.address[offset + index] = sub / mBatchSize;
				}


				loss += mLossTblOnCPU.address[batchID] = result;
			}

#if _DEBUG
			assert(mBatchSize != 0);
#endif
			return loss / mBatchSize;
		}
	}
}