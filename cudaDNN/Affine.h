#pragma once
#include "BaseLayer.h"

namespace miduho
{
	namespace layer
	{

		class Affine : public BaseLayer
		{
		public:
			void initialize();
			void forward(void*);
			void backward(void*);
			void memcpyHostToDevice();
			void memcpyDeviceToHost();

		private:
			void forwardOnGPU();
			void forwardOnCPU();
			void backwardOnGPU();
			void backwardOnCPU();
		};

	}
}