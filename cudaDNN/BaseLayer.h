#pragma once
#include <vector>


namespace miduho
{
	namespace layer
	{

		class BaseLayer
		{
		public:
			BaseLayer() = default;
			virtual ~BaseLayer() = default;

			virtual void initialize() = 0;
			virtual void forward(void*) = 0;
			virtual void backward(void*) = 0;
			virtual void memcpyHostToDevice() = 0;
			virtual void memcpyDeviceToHost() = 0;
		protected:
			virtual void forwardOnGPU() = 0;
			virtual void forwardOnCPU() = 0;
			virtual void backwardOnGPU() = 0;
			virtual void backwardOnCPU() = 0;
		};

		class TestLayer : public BaseLayer
		{
		public:
			TestLayer() {}
			TestLayer(int) {}

			~TestLayer() {}

			void initialize() {}
			void forward(void*) {}
			void backward(void*) {}

		protected:
			void memcpyHostToDevice() {}
			void memcpyDeviceToHost() {}
			void forwardOnGPU()	{}
			void forwardOnCPU()	{}
			void backwardOnGPU(){}
			void backwardOnCPU(){}
		};

	}
}