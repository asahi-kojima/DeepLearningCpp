#pragma once

#include "../setting.h"
#include <cassert>

//Timeデバッグは以下のIndexデバッグと併用すると
//正確な値が出ないので注意。
#define TIME_DEBUG (1 & _DEBUG)
#if TIME_DEBUG
#include <map>
#include <string>
extern std::map<std::string, f32> timers;
#endif

#define INDEX_DEBUG (0 & _DEBUG)

#define ON_CPU_DEBUG (1)

#define GPU_SYNC_DEBUG (1 & _DEBUG)

namespace Aoba
{
	using cf32 = const f32;


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


	struct DataArray
	{
		f32* address;
		u32 size;
		u32 byteSize;

		f32& operator[](u32 index)
		{
#if INDEX_DEBUG
			if (index >= size)
			{
				assert(0);
			}
#endif
			return this->address[index];
		}



		//4Dの配列として使うための
	private:
		u32 mBatchSize;
		u32 mChannel;
		u32 mHeight;
		u32 mWidth;

		u32 mCHW;
		u32 mHW;

		bool mIsSetSize = false;

		/////////////////////////////////////////////////////////////////
		// 4D
		/////////////////////////////////////////////////////////////////
	private:
		bool mIs4DArray = false;
	public:
		bool is4DArray() { return mIs4DArray; }
		void setSizeAs4D(u32 batch, u32 c, u32 h, u32 w)
		{
			if (mIsSetSize)
			{
				assert(0);
			}
			mIsSetSize = true;


			mIs4DArray = true;


			mBatchSize = batch;
			mChannel = c;
			mHeight = h;
			mWidth = w;


			mHW = mHeight * mWidth;
			mCHW = mChannel * mHW;


			size = mBatchSize * mChannel * mHeight * mWidth;
			byteSize = sizeof(f32) * size;
		}

		void setSizeAs4D(u32 batch, const DataShape& dataShape)
		{
			setSizeAs4D(batch, dataShape.channel, dataShape.height, dataShape.width);
		}






		/////////////////////////////////////////////////////////////////
		// 3D
		/////////////////////////////////////////////////////////////////
	private:
		bool mIs3DArray = false;
	public:
		void setSizeAs3D(u32 batchSize, u32 h, u32 w)
		{
			if (mIsSetSize)
			{
				assert(0);
			}
			mIsSetSize = true;


			mIs3DArray = true;


			mBatchSize = batchSize;
			mChannel = 0;
			mHeight = h;
			mWidth = w;


			mHW = mHeight * mWidth;
			mCHW = mChannel * mHW;


			size = mBatchSize * mHeight * mWidth;
			byteSize = sizeof(f32) * size;
		}



		/////////////////////////////////////////////////////////////////
		// 2D
		/////////////////////////////////////////////////////////////////
	private:
		bool mIs2DArray = false;
	public:
		void setSizeAs2D(u32 batchSize, u32 eachSize)
		{
			if (mIsSetSize)
			{
				assert(0);
			}
			mIsSetSize = true;


			mIs2DArray = true;


			mBatchSize = batchSize;
			mChannel = 0;
			mHeight = 0;
			mWidth = eachSize;

			size = mBatchSize * mWidth;
			byteSize = sizeof(f32) * size;
		}




		////////////////////////////////////////////////
		f32& operator()(u32 N, u32 C, u32 H, u32 W)
		{
			u32 index = N * mCHW + C * mHW + H * mWidth + W;
#if INDEX_DEBUG
			if (!mIs4DArray)
			{
				assert(0);
			}
			if (N >= mBatchSize || C >= mChannel || H >= mHeight || W >= mWidth)
			{
				assert(0);
			}
			if (index >= size)
			{
				assert(0);
			}
#endif
			return this->address[index];
		}


		f32& operator()(u32 N, u32 H, u32 W)
		{
			if (mIs4DArray)
			{
				u32 c = H;
				u32 hw = W;
				u32 index = N * mCHW + c * mHW + hw;
#if INDEX_DEBUG
				if (N >= mBatchSize || c >= mChannel || hw >= mHW)
				{
					assert(0);
				}
				if (index >= size)
				{
					assert(0);
				}
#endif
				return this->address[index];
			}

			if (mIs3DArray)
			{
				u32 index = N * mHW + H * mWidth + W;
#if INDEX_DEBUG
				if (N >= mBatchSize || H >= mHeight || W >= mWidth)
				{
					assert(0);
				}
				if (index >= size)
				{
					assert(0);
				}
#endif
				return this->address[index];
			}

			assert(0);
		}

		f32& operator()(u32 N, u32 W)
		{
			if (mIs4DArray)
			{
				u32 index = N * mCHW + W;
#if INDEX_DEBUG
				if (N >= mBatchSize || W >= mCHW)
				{
					assert(0);
				}
				if (index >= size)
				{
					assert(0);
				}
#endif
				return this->address[index];
			}

			if (mIs3DArray)
			{
				u32 index = N * mHW + W;
#if INDEX_DEBUG
				if (N >= mBatchSize || W >= mHW)
				{
					assert(0);
				}
				if (index >= size)
				{
					assert(0);
				}
#endif
				return this->address[index];
			}

			if (mIs2DArray)
			{

				u32 index = N * mWidth + W;
#if INDEX_DEBUG
				if (N >= mBatchSize || W >= mWidth)
				{
					assert(0);
				}
				if (index >= size)
				{
					assert(0);
				}
#endif
				return this->address[index];
			}
		}
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
			eachTrainingDataSize = trainingDataShape.getDataSize();
			eachCorrectDataSize = correctDataShape.getDataSize();
		}

		DataFormat4DeepLearning() = default;
	};
}