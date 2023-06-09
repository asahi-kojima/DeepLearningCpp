#include "L2Loss.h"


namespace Aoba
{
	namespace lossFunction
	{
		void L2Loss::mallocOnCPU()
		{
			mDInputDataOnCPU.setSizeAs4D(mBatchSize, mTrainingDataShape);
			MALLOC_AND_INITIALIZE_0_ON_CPU(mDInputDataOnCPU);

			mLossTblOnCPU.size = mBatchSize;
			MALLOC_AND_INITIALIZE_0_ON_CPU(mLossTblOnCPU);
		}

		f32 L2Loss::calcLossAndDInputOnCPU()
		{
			DataArray& input = *mForwardResultOnCPU;
			DataArray& correctData = *mCorrectDataOnCPU;

			const u32 dataSize = input.size / mBatchSize;

			f32 loss = 0.0f;

			if (mTrainingDataShape != mCorrectDataShape)
			{
				std::cout << "shape mismatch" << std::endl;
				assert(0);
			}

			for (u32 batchID = 0; batchID < mBatchSize; batchID++)
			{
				const u32 offset = batchID * dataSize;

				f32 result = 0.0f;
				for (u32 index = 0; index < dataSize; index++)
				{
					const f32 diff = input[offset + index] - correctData[offset + index];
					result += 0.5 * diff * diff;

					mDInputDataOnCPU[offset + index] = diff;// / mBatchSize;
				}
				//ƒeƒXƒgŽÀ‘•
				//constexpr f32 ep = 1e-7;
				//for (u32 index = 0; index < dataSize; index++)
				//{
				//	const f32 correct = correctData[offset + index] - ep;
				//	const f32 diff = (input[offset + index] - correct) / correct;
				//	result += 0.5 * diff * diff;

				//	mDInputDataOnCPU[offset + index] = diff;// / mBatchSize;
				//}

				loss += mLossTblOnCPU[batchID] = result;
			}


			return loss / mBatchSize;
		}
	}
}