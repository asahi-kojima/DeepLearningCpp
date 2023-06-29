#pragma once

//Time�f�o�b�O�͈ȉ���Index�f�o�b�O�ƕ��p�����
//���m�Ȓl���o�Ȃ��̂Œ��ӁB
//GPU�g�p����GPUSync���؂�Ă����Time�f�o�b�O��
//���m�Ȓl���o�Ȃ��̂Œ��ӁB
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