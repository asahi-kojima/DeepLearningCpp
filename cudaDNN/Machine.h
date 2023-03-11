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
		using flowDataType = layer::BaseLayer::flowDataType;
		
		Machine() = default;
		~Machine() = default;



		/// <summary>
		/// ƒf[ƒ^‚Ì€”õ
		/// Layer‚Ì€”õ
		/// </summary>
		/// <returns></returns>
		bool initialize();

		bool preProcess();
		bool mainProcess();
		bool postProcess();

		/// <summary>
		/// Layer‚Ì”pŠü
		/// </summary>
		/// <returns></returns>
		bool terminate();

		void initializeLayer();
		void setupLayer();

		std::vector<std::unique_ptr<layer::BaseLayer> > mLayerList;
		layer::BaseLayer::flowDataFormat mFlowData = layer::BaseLayer::flowDataFormat{100, 1, 1, 100};
	};
}