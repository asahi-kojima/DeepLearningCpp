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
	};

	struct DataFormat4DeepLearning
	{
		//データの総数
		u32 dataNum;

		//計算時のバッチ数
		u32 batchSize = 100;

		//訓練データの形状
		DataShape trainingDataShape;
		//個々の訓練データの要素数
		u32 eachTrainingDataSize;

		//教師データの形状
		DataShape correctDataShape;
		//個々の教師データの要素数
		u32 eachCorrectDataSize;

		DataFormat4DeepLearning(u32 dataNum, u32 batchSize, DataShape trainingDataShape, DataShape correctDataShape)
			: dataNum(dataNum)
			, batchSize(batchSize)
			, trainingDataShape(trainingDataShape)
			, correctDataShape(correctDataShape)
		{
			eachTrainingDataSize = trainingDataShape.channel * trainingDataShape.height * trainingDataShape.width;
			eachCorrectDataSize = correctDataShape.channel * correctDataShape.height * correctDataShape.width;
		}

		DataFormat4DeepLearning() = default;
	};
}