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


	//struct DataMemory
	//{
	//	cf32* address;
	//	u32 size;
	//	u32 byteSize;
	//	DataMemory(DataMemory data)
	//		:address(data.address)
	//		,size(data.size)
	//		,byteSize(data.byteSize)
	//	{
	//	}
	//};

	struct paramMemory
	{
		f32* address;
		u32 size;
	};
}