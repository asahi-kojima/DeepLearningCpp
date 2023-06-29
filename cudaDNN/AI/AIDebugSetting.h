#pragma once

//Timeデバッグは以下のIndexデバッグと併用すると
//正確な値が出ないので注意。
//GPU使用時にGPUSyncが切れているとTimeデバッグは
//正確な値が出ないので注意。
#define TIME_DEBUG (1 & _DEBUG)
#if TIME_DEBUG
#include <map>
#include <string>
#include <chrono>
extern std::map<std::string, f32> timers;
#endif

#define INDEX_DEBUG (0 & _DEBUG)

#define ON_CPU_DEBUG (0)

#define GPU_SYNC_DEBUG (1 & _DEBUG)