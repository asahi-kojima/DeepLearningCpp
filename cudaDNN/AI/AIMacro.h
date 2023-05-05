#pragma once


#define FORWARD_ON_(Device)											\
for (auto& layer : mLayerList)										\
{																	\
	layer->forwardOn##Device();										\
}																	\
																	\
mLossOn##Device = mLossFunction->calcLossAndDInputOn##Device();





#define BACKWARD_ON_(Device)																\
for (auto riter = mLayerList.rbegin(), end = mLayerList.rend(); riter != end; riter++)		\
{																							\
	(*riter)->backwardOn##Device();															\
}





#define OPTIMIZE_ON_(Device)					\
for (auto& layer : mLayerList)					\
{												\
	mOptimizer->optimizeOn##Device(layer);		\
}





#define INITIALIZE_ON_(Device)																																	\
DataShape shape = mDataFormat4DeepLearning.trainingDataShape;																									\
for (auto& layer : mLayerList)																																	\
{																																								\
	layer->initializeOn##Device(mDataFormat4DeepLearning.batchSize, shape);																						\
}																																								\
																																								\
mLossFunction->initializeOn##Device(mDataFormat4DeepLearning.batchSize,mDataFormat4DeepLearning.trainingDataShape,mDataFormat4DeepLearning.correctDataShape);	\
																																								\
DataMemory* pInputDataOn##Device = &mInputTrainingDataOn##Device;																								\
																																								\
for (auto& layer : mLayerList)																																	\
{																																								\
	layer->setInputDataOn##Device(pInputDataOn##Device);																										\
}																																								\
																																								\
mForwardResultOn##Device = pInputDataOn##Device;																												\
																																								\
mLossFunction->setInputOn##Device(mForwardResultOn##Device, &mInputCorrectDataOn##Device);																		\
																																								\
																																								\
DataMemory* pDInputDataOn##Device = mLossFunction->getDInputDataOn##Device();																					\
																																								\
for (auto rit = mLayerList.rbegin(); rit != mLayerList.rend(); rit++)																							\
{																																								\
	(*rit)->setDInputDataOn##Device(pDInputDataOn##Device);																										\
}