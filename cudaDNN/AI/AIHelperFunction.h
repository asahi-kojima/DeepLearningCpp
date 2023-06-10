#pragma once

#include <vector>
#include <cuda_runtime.h>

#include "../setting.h"
#include "AISetting.h"
#include "AIMacro.h"

namespace Aoba
{
	std::vector<f32> cpuDataFromGpuData(DataArray& data)
	{
		std::vector<f32> v;
		v.resize(data.size);

		CHECK(cudaMemcpy(v.data(), data.address, data.size * sizeof(f32), cudaMemcpyDeviceToHost));
		return v;
		return std::move(v);
	}
}