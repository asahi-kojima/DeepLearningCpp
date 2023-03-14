#pragma once



#define SINGLETON(CLASSNAME) \
private:\
	CLASSNAME() = default;\
	~CLASSNAME() = default;\
public:\
	CLASSNAME(const CLASSNAME&) = delete;\
	CLASSNAME& operator=(const CLASSNAME&) = delete;\
	CLASSNAME(CLASSNAME&&) = delete;\
	CLASSNAME& operator=(CLASSNAME&&) = delete;\
	static CLASSNAME& getInstance()\
	{\
		static CLASSNAME instance;\
		return instance;\
	}

namespace miduho
{
	namespace layer
	{
		class BaseLayer;
	}
	namespace optimizer
	{
		class BaseOptimizer
		{
		public:
			BaseOptimizer() = default;
			~BaseOptimizer() = default;
			virtual void optimize(layer::BaseLayer*) = 0;
		};
	}
}