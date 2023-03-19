#pragma once

#include "../setting.h"

namespace Aoba
{
	using parameterType = f32;
	using flowDataType = f32;

	struct DataMemory
	{
		f32* address;
		u32 size;
	};

	using constDataMemory = const DataMemory;

	struct paramMemory
	{
		f32* address;
		u32 size;
	};
}