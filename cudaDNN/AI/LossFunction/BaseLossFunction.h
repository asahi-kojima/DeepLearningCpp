#pragma once
#include "../AIDataStructure.h"
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
			void setInputOnCPU(DataArray* pData, DataArray* pCorrectData)
			{
				mForwardResultOnCPU = pData;
				mCorrectDataOnCPU = pCorrectData;
			}
			DataArray* getDInputDataOnCPU()
			{
				return &mDInputDataOnCPU;
			}
			virtual void terminateOnCPU() = 0;


			virtual f32 calcLossAndDInputOnGPU() = 0;
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
			void setInputOnGPU(DataArray* pData, DataArray* pCorrectData)
			{
				mForwardResultOnGPU = pData;
				mCorrectDataOnGPU = pCorrectData;
			}
			DataArray* getDInputDataOnGPU()
			{
				return &mDInputDataOnGPU;
			}
			virtual void terminateOnGPU() = 0;

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
			DataArray mLossTblOnCPU;
			DataArray mDInputDataOnCPU;
			DataArray* mForwardResultOnCPU;
			DataArray* mCorrectDataOnCPU;

			virtual void mallocOnCPU() = 0;

			//////////////////////////////////////////////////////////////////////
			//GPU�֌W�̕ϐ�
			//////////////////////////////////////////////////////////////////////
			DataArray mLossTblOnGPU;
			DataArray mDInputDataOnGPU;
			DataArray* mForwardResultOnGPU;
			DataArray* mCorrectDataOnGPU;

			virtual void mallocOnGPU() = 0;

		private:
			bool mIsSetupDataShape = false;
		};
	}
}