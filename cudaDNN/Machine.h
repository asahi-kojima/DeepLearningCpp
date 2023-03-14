#pragma once
#include <vector>
#include <memory>

#include "setting.h"
#include "./Layer/BaseLayer.h"


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
		using dataMemory = layer::BaseLayer::DataMemory;

		struct learningDataSet
		{
			dataMemory allInputData;
			dataMemory allTrainingData;
			dataMemory InputData;
			dataMemory TrainingData;
			u32 inputDataOffset;
			u32 trainingDataOffset;
			layer::BaseLayer::FlowDataFormat dataShape =
				layer::BaseLayer::FlowDataFormat{ 100, 1, 1, 28 * 28 };
		};


		Machine() = default;
		~Machine() = default;


		//‘å˜g
		//(1)
		bool initialize();
		//(2)
		bool preProcess();
		//(3)
		bool mainProcess();
		//(4)
		bool postProcess();
		//(5)
		bool terminate();

		//(1)‚ÅŒÄ‚Î‚ê‚éŠÖ”
		void entryLayer(std::unique_ptr<layer::BaseLayer>&&);
		void setupLayerInfo();
		void initializeLayer();

		//(3)‚ÅŒÄ‚Î‚ê‚éŠÖ”
		void startLearning();
		void makeTestData();


		std::vector<std::unique_ptr<layer::BaseLayer> > mLayerList;
		learningDataSet mLearningData;
	};
}