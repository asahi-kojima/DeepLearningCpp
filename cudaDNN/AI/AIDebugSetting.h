#pragma once

#define ON 1
#define OFF 0

//Time�f�o�b�O�͈ȉ���Index�f�o�b�O�ƕ��p�����
//���m�Ȓl���o�Ȃ��̂Œ��ӁB
//GPU�g�p����GPUSync���؂�Ă����Time�f�o�b�O��
//���m�Ȓl���o�Ȃ��̂Œ��ӁB
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