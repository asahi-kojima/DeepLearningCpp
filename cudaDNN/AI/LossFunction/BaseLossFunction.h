#pragma once

#include "../AISetting.h"
#include "../Layer/BaseLayer.h"
namespace Aoba
{
	namespace lossFunction
	{
		class BaseLossFunction
		{
		public:
			BaseLossFunction() = default;
			~BaseLossFunction() = default;


			virtual f32 calcLossAndDInputOnCPU() = 0;
			virtual void initializeOnCPU(u32 batchSize, DataShape& trainingDataShape, DataShape& correctDataShape) final
			{
				if (!mIsSetupDataShape)
				{
					mBatchSize = batchSize;
					mTrainingDataShape = trainingDataShape;
					mCorrectDataShape = correctDataShape;
					mIsSetupDataShape = true;
				}
				mallocOnCPU();
			}
			void setInputOnCPU(DataMemory* pData, DataMemory* pCorrectData)
			{
				mForwardResultOnCPU = pData;
				mCorrectDataOnCPU = pCorrectData;
			}
			DataMemory* getDInputDataOnCPU()
			{
				return &mDInputDataOnCPU;
			}



			virtual void initializeOnGPU(u32 batchSize, DataShape& trainingDataShape, DataShape& correctDataShape) final
			{
				if (!mIsSetupDataShape)
				{
					mBatchSize = batchSize;
					mTrainingDataShape = trainingDataShape;
					mCorrectDataShape = correctDataShape;
					mIsSetupDataShape = true;
				}
				mallocOnGPU();
			}
			virtual f32 calcLossAndDInputOnGPU() = 0;
			void setInputOnGPU(DataMemory* pData, DataMemory* pCorrectData)
			{
				mForwardResultOnGPU = pData;
				mCorrectDataOnGPU = pCorrectData;
			}

			DataMemory* getDInputDataOnGPU()
			{
				return &mDInputDataOnGPU;
			}

		protected:
			//////////////////////////////////////////////////////////////////////
			//���ʕϐ�
			//////////////////////////////////////////////////////////////////////
			//���̕ϐ��͋��t�f�[�^�̃f�[�^�`��
			u32 mBatchSize;
			DataShape mTrainingDataShape;
			DataShape mCorrectDataShape;

			//////////////////////////////////////////////////////////////////////
			//CPU�֌W�̕ϐ�
			//////////////////////////////////////////////////////////////////////
			DataMemory mLossTblOnCPU;
			DataMemory mDInputDataOnCPU;
			DataMemory* mForwardResultOnCPU;
			DataMemory* mCorrectDataOnCPU;

			virtual void mallocOnCPU() = 0;

			//////////////////////////////////////////////////////////////////////
			//GPU�֌W�̕ϐ�
			//////////////////////////////////////////////////////////////////////
			DataMemory mLossTblOnGPU;
			DataMemory mDInputDataOnGPU;
			DataMemory* mForwardResultOnGPU;
			DataMemory* mCorrectDataOnGPU;

			virtual void mallocOnGPU() = 0;

		private:
			bool mIsSetupDataShape = false;
		};
	}
}