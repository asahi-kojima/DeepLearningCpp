#pragma once

#define ON 1
#define OFF 0

//Timeデバッグは以下のIndexデバッグと併用すると
//正確な値が出ないので注意。
//GPU使用時にGPUSyncが切れているとTimeデバッグは
//正確な値が出ないので注意。
#define TIME_DEBUG (ON & _DEBUG)
#if TIME_DEBUG
#include <map>
#include <string>
#include <chrono>
extern std::map<std::string, f32> timers;
#endif

#define INDEX_DEBUG (OFF & _DEBUG)

#define ON_CPU_DEBUG (OFF)

#define GPU_SYNC_DEBUG (ON & _DEBUG)