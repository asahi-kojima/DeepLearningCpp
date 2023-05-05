#pragma once

#include "../setting.h"

namespace Aoba
{
	using cf32 = const f32;

	struct DataMemory
	{
		f32* address;
		u32 size;
		u32 byteSize;
	};

	struct paramMemory
	{
		f32* address;
		u32 size;
		u32 byteSize;
	};

	struct DataShape
	{
		u32 channel;
		u32 height;
		u32 width;

		u32 getDataSize() const
		{
			return channel * height * width;
		}

		bool operator==(const DataShape& comp) const
		{
			return ((this->channel == comp.channel)
				&& (this->height == comp.height)
				&& (this->width == comp.width));
		}

		bool operator!=(const DataShape& comp) const
		{
			return !((*this) == comp);
		}
	};

	struct DataFormat4DeepLearning
	{
		//�f�[�^�̑���
		u32 dataNum;

		//�v�Z���̃o�b�`��
		u32 batchSize = 100;

		//�P���f�[�^�̌`��
		DataShape trainingDataShape;
		//�X�̌P���f�[�^�̗v�f��
		u32 eachTrainingDataSize;

		//���t�f�[�^�̌`��
		DataShape correctDataShape;
		//�X�̋��t�f�[�^�̗v�f��
		u32 eachCorrectDataSize;

		DataFormat4DeepLearning(u32 dataNum, u32 batchSize, DataShape trainingDataShape, DataShape correctDataShape)
			: dataNum(dataNum)
			, batchSize(batchSize)
			, trainingDataShape(trainingDataShape)
			, correctDataShape(correctDataShape)
		{
			eachTrainingDataSize = trainingDataShape.getDataSize();
			eachCorrectDataSize = correctDataShape.getDataSize();
		}

		DataFormat4DeepLearning() = default;
	};
}