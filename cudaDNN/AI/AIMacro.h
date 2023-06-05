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
mLossFunction->initializeOn##Device(mDataFormat4DeepLearning.batchSize,shape,mDataFormat4DeepLearning.correctDataShape);	\
																																								\
DataArray* pInputDataOn##Device = &mInputTrainingDataOn##Device;																								\
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
DataArray* pDInputDataOn##Device = mLossFunction->getDInputDataOn##Device();																					\
																																								\
for (auto rit = mLayerList.rbegin(); rit != mLayerList.rend(); rit++)																							\
{																																								\
	(*rit)->setDInputDataOn##Device(pDInputDataOn##Device);																										\
}

#define MALLOC_ON_CPU(memory)              \
{                                          \
    memory.address = new f32[memory.size]; \
}

#define MALLOC_ON_GPU(memory)                                                    \
{                                                                                \
    CHECK(cudaMalloc((void**)(&(memory.address)), memory.size * sizeof(f32)));   \
}



#define INITIALIZE_CPU_DATA_0(memory)                                                                            \
{                                                                                                                \
    {                                                                                                            \
        for (u32 idx = 0; idx < memory.size; idx++)                                                              \
        {                                                                                                        \
            memory.address[idx] = 0.0f;                                                                          \
        }                                                                                                        \
    }                                                                                                            \
}

#define MALLOC_AND_INITIALIZE_0_ON_CPU(memory)  \
{                                               \
    {                                           \
        MALLOC_ON_CPU(memory);                  \
        INITIALIZE_CPU_DATA_0(memory);          \
    }                                           \
}

#define INITIALIZE_CPU_DATA_1(memory)                                                                            \
{                                                                                                                \
    {                                                                                                            \
        for (u32 idx = 0; idx < memory.size; idx++)                                                              \
        {                                                                                                        \
            memory.address[idx] = 1.0f;                                                                          \
        }                                                                                                        \
    }                                                                                                            \
}

#define MALLOC_AND_INITIALIZE_1_ON_CPU(memory)  \
{                                               \
    {                                           \
        MALLOC_ON_CPU(memory);                  \
        INITIALIZE_CPU_DATA_1(memory);          \
    }                                           \
}

#define INITIALIZE_CPU_DATA_NORMAL(memory, elements, weight)                                               \
{                                                                                                          \
    {                                                                                                      \
        std::random_device seed_gen;                                                                       \
        std::default_random_engine engine(seed_gen());                                                     \
        std::normal_distribution<> dist(0.0, std::sqrt(2.0 / elements));                                   \
        for (u32 idx = 0; idx < memory.size; idx++)                                                        \
        {                                                                                                  \
            memory.address[idx] = weight * static_cast<f32>(dist(engine));                                 \
        }                                                                                                  \
    }                                                                                                      \
}

#define MALLOC_AND_INITIALIZE_NORMAL_ON_CPU(memory,elements, weight)   \
{                                                                      \
    {                                                                  \
        MALLOC_ON_CPU(memory);                                         \
        INITIALIZE_CPU_DATA_NORMAL(memory, elements, weight);          \
    }                                                                  \
}

#define INITIALIZE_GPU_DATA_0(memory)                                                                            \
{                                                                                                                \
    {                                                                                                            \
        std::vector<f32> tmp(memory.size);                                                                       \
        for (u32 idx = 0; idx < memory.size; idx++)                                                              \
        {                                                                                                        \
            tmp[idx] = 0.0f;                                                                                     \
        }                                                                                                        \
        CHECK(cudaMemcpy(memory.address, tmp.data(), memory.size * sizeof(f32), cudaMemcpyHostToDevice));        \
    }                                                                                                            \
}
#define INITIALIZE_GPU_DATA_1(memory)                                                                            \
{                                                                                                                \
    {                                                                                                            \
        std::vector<f32> tmp(memory.size);                                                                       \
        for (u32 idx = 0; idx < memory.size; idx++)                                                              \
        {                                                                                                        \
            tmp[idx] = 1.0f;                                                                                     \
        }                                                                                                        \
        CHECK(cudaMemcpy(memory.address, tmp.data(), memory.size * sizeof(f32), cudaMemcpyHostToDevice));        \
    }                                                                                                            \
}
#define INITIALIZE_GPU_DATA_NORMAL(memory, elements, weight)                                               \
{                                                                                                          \
    {                                                                                                      \
        std::random_device seed_gen;                                                                       \
        std::default_random_engine engine(seed_gen());                                                     \
        std::normal_distribution<> dist(0.0, std::sqrt(2.0 / elements));                                   \
        std::vector<f32> tmp(memory.size);                                                                 \
        for (u32 idx = 0; idx < memory.size; idx++)                                                        \
        {                                                                                                  \
            tmp[idx] = weight * static_cast<f32>(dist(engine));                                            \
        }                                                                                                  \
        CHECK(cudaMemcpy(memory.address, tmp.data(), memory.size * sizeof(f32), cudaMemcpyHostToDevice));  \
    }                                                                                                      \
}


inline void mallocCPUData(Aoba::DataArray& memory)
{
    memory.address = new f32[memory.size];
}

inline void initializeDataOnCPU_0(const Aoba::DataArray& memory)
{
	for (u32 idx = 0; idx < memory.size; idx++)
	{
		memory.address[idx] = 0.0f;
	}
}

inline void initializeDataOnCPU_1(const Aoba::DataArray& memory)
{
    for (u32 idx = 0; idx < memory.size; idx++)
    {
        memory.address[idx] = 1.0f;
    }
}