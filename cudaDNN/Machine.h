#pragma once
#include <vector>
#include <memory>

#include "setting.h"
#include "BaseLayer.h"


namespace miduho
{
	class Machine
	{
	public:
		static Machine& getInstance()
		{
			static Machine instance;
			return instance;
		}

		u32 entry();

	private:
		Machine() = default;
		~Machine() = default;

		bool initialize();
		bool preProcess();
		bool mainProcess();
		bool postProcess();
		bool terminate();

		std::vector<std::unique_ptr<layer::BaseLayer> > mLayerList;
	};
}