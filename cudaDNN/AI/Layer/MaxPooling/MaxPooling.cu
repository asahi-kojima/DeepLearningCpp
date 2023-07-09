#include <random>
#include <cuda_runtime.h>
#include <cassert>
#include <device_functions.h>

//このマクロはCUDAファイルがコンパイルされる時に定義される。
//インテリセンスのエラーを一時的に抑制するためにこの定義を置いている。
#if !defined(__CUDACC__)
#define __CUDACC__
#endif

#include "../../AIHelperFunction.h"
#include "../../AIMacro.h"
#include "MaxPooling.h"


namespace Aoba {
	namespace layer
	{
		namespace
		{
			__device__ bool rangeChecker(s32 index, u32 maxIndex)
			{
				return (0 <= index && index < maxIndex);
			}

			__device__ bool rangeCheckerHW(s32 indexH, s32 indexW, u32 maxIndexH, u32 maxIndexW)
			{
				return rangeChecker(indexH, maxIndexH) && rangeChecker(indexW, maxIndexW);
			}

			__global__ void cudaPoolingForward(
				f32* y, f32* input, s32* poolingMask,
				MaxPooling::parameterInfo* pInfo, u32 batchSize, u32 mOcOhOw)
			{
				MaxPooling::parameterInfo info = *pInfo;

				const u32 N = blockIdx.x * blockDim.x + threadIdx.x;
				const u32 OcOhOw = blockIdx.y * blockDim.y + threadIdx.y;

				if (N >= batchSize || OcOhOw >= mOcOhOw)
				{
					return;
				}

				const u32 mIh = info.Ih;
				const u32 mIw = info.Iw;
				const u32 mIhIw = mIh * mIw;
				const u32 mIcIhIw = info.IcIhIw;
				const u32 mOw = info.Ow;
				const u32 mOhOw = info.OhOw;
				const u32 Oc = OcOhOw / mOhOw;
				const u32 Ic = Oc;
				const u32 OhOw = OcOhOw - Oc * mOhOw;
				const u32 Oh = OhOw / mOw;
				const u32 Ow = OhOw - Oh * mOw;


				const u32 mSh = info.Sh;
				const u32 mSw = info.Sw;
				const u32 mPh = info.Ph;
				const u32 mPw = info.Pw;
				const s32 basisIndexIh = Oh * mSh - mPh;
				const s32 basisIndexIw = Ow * mSw - mPw;

				s32 maxCandIh = basisIndexIh;
				s32 maxCandIw = basisIndexIw;
				f32 maxValueCand = rangeCheckerHW(maxCandIh, maxCandIw, mIh, mIw) ?
					input[N * mIcIhIw + Ic * mIhIw + maxCandIh * mIw + maxCandIw] : 0;

				const u32 mFh = info.Fh;
				const u32 mFw = info.Fw;

				for (u32 fh = 0; fh < mFh; fh++)
				{
					for (u32 fw = 0; fw < mFw; fw++)
					{
						const s32 indexIh = basisIndexIh + fh;
						const s32 indexIw = basisIndexIw + fw;

						const f32 value = rangeCheckerHW(indexIh, indexIw, mIh, mIw) ?
							input[N * mIcIhIw + Ic * mIhIw + indexIh * mIw + indexIw] : 0;

						if (maxValueCand < value)
						{
							maxValueCand = value;
							maxCandIh = indexIh;
							maxCandIw = indexIw;
						}
					}
				}

				poolingMask[N * mOcOhOw + OcOhOw] = rangeCheckerHW(maxCandIh, maxCandIw, mIh, mIw) ? Ic * mIhIw + maxCandIh * mIw + maxCandIw : -1;
				y[N * mOcOhOw + OcOhOw] = maxValueCand;
			}

			__global__ void backwardOnGPU_init(
				f32* backwardResult,
				u32 batchSize,
				u32 mIcIhIw)
			{
				u32 N = blockIdx.x * blockDim.x + threadIdx.x;//input
				u32 IcIhIw = blockIdx.y * blockDim.y + threadIdx.y;//batch

				if (N >= batchSize || IcIhIw >= mIcIhIw)
				{
					return;
				}

				u32 id = N * mIcIhIw + IcIhIw;

				
				backwardResult[id] = 0;
			}

			__global__ void backwardOnGPU_dout(
				f32* backwardResult,
				f32* dout,
				s32* poolingMask,
				u32 batchSize,
				u32 mOcOhOw,
				u32 mIcIhIw)
			{
				u32 N = blockIdx.x * blockDim.x + threadIdx.x;//
				u32 OcOhOw = blockIdx.y * blockDim.y + threadIdx.y;//

				if (N >= batchSize || OcOhOw >= mOcOhOw)
				{
					return;
				}

				u32 id = N * mOcOhOw + OcOhOw;

				s32 index = poolingMask[id];

				if (index < 0)
				{
					return;
				}

				//backwardResult[N * mIcIhIw + index] += dout[id];
				atomicAdd(&(backwardResult[N * mIcIhIw + index]), dout[id]);
			}
		}



		void MaxPooling::mallocOnGPU()
		{
			//Poolingマスクの領域確保
			CHECK(cudaMalloc((void**)(&mPoolingMaskOnGPU), mBatchSize * mOcOhOw * sizeof(s32)));

			//計算結果を格納するためのメモリ確保
			mForwardResultOnGPU.setSizeAs4D(mBatchSize, mOc, mOh, mOw);
			mBackwardResultOnGPU.setSizeAs4D(mBatchSize, mIc, mIh, mIw);
			MALLOC_AND_INITIALIZE_0_ON_GPU(mForwardResultOnGPU);
			MALLOC_AND_INITIALIZE_0_ON_GPU(mBackwardResultOnGPU);

			//GPUに渡すパラメータ
			parameterInfo tmp;
			tmp.batchSize = mBatchSize;
			tmp.Ic = mIc;
			tmp.Ih = mIh;
			tmp.Iw = mIw;
			tmp.IcIhIw = mIcIhIw;
			tmp.IcFhFw = mIcFhFw;
			tmp.Oc = mOc;
			tmp.Oh = mOh;
			tmp.Ow = mOw;
			tmp.OhOw = mOhOw;
			tmp.OcOhOw = mOc * mOhOw;
			tmp.FhFw = mFhFw;
			tmp.IhIw = mIh * mIw;
			tmp.Fn = mOc;
			tmp.Fh = mFh;
			tmp.Fw = mFw;
			tmp.Sh = mSh;
			tmp.Sw = mSw;
			tmp.Ph = mPh;
			tmp.Pw = mPw;
			CHECK(cudaMalloc(&mParameterInfoOnGPU, sizeof(parameterInfo)););
			CHECK(cudaMemcpy(mParameterInfoOnGPU, &tmp, sizeof(parameterInfo), cudaMemcpyHostToDevice));
		}

		void MaxPooling::forwardOnGPU()
		{
			auto& input = *mInputDataOnGPU;

			{
				dim3 block(16, 16);
				dim3 grid(
					(mBatchSize + block.x - 1) / block.x,
					(mOcOhOw + block.y - 1) / block.y);
#if TIME_DEBUG
				{
					std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
					cudaPoolingForward << <grid, block >> >
						(mForwardResultOnGPU.address,
							input.address,
							mPoolingMaskOnGPU,
							mParameterInfoOnGPU,
							mBatchSize,
							mOcOhOw);

#if GPU_SYNC_DEBUG
					CHECK(cudaDeviceSynchronize());
#endif
#if TIME_DEBUG
					f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
					std::string name = "";
					(((name += __FUNCTION__) += " : ") += std::to_string(mInstanceID)) += " : cudaCreateReshapedData";
					timers[name] = elapsedTime;
				}
#endif
			}
		}

		void MaxPooling::backwardOnGPU()
		{
			auto& dout = *mDInputDataOnGPU;

			//逆伝搬の出力を0初期化
			{
				dim3 block(16, 16);
				dim3 grid(
					(mBatchSize + block.x - 1) / block.x,
					(mIcIhIw + block.y - 1) / block.y);
#if TIME_DEBUG
				{
					std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
					backwardOnGPU_init << <grid, block >> > (
						mBackwardResultOnGPU.address,
						mBatchSize, mIcIhIw);

#if GPU_SYNC_DEBUG
					CHECK(cudaDeviceSynchronize());
#endif
#if TIME_DEBUG
					f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
					std::string name = "";
					(((name += __FUNCTION__) += " : ") += std::to_string(mInstanceID)) += " : backwardOnGPU_init";
					timers[name] = elapsedTime;
				}
#endif
			}


			//逆伝搬の出力
			{
				dim3 block(16, 16);
				dim3 grid(
					(mBatchSize + block.x - 1) / block.x,
					(mOcOhOw + block.y - 1) / block.y);
#if TIME_DEBUG
				{
					std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
					backwardOnGPU_dout << <grid, block >> > (
						mBackwardResultOnGPU.address,
						dout.address,
						mPoolingMaskOnGPU,
						mBatchSize, mOcOhOw, mIcIhIw);

#if GPU_SYNC_DEBUG
					CHECK(cudaDeviceSynchronize());
#endif
#if TIME_DEBUG
					f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
					std::string name = "";
					(((name += __FUNCTION__) += " : ") += std::to_string(mInstanceID)) += " : backwardOnGPU_dout";
					timers[name] = elapsedTime;
				}
#endif
			}
		}

		void MaxPooling::terminateOnGPU()
		{
			CUDA_FREE(mForwardResultOnGPU);
			CUDA_FREE(mBackwardResultOnGPU);

			cudaFree(mPoolingMaskOnGPU);
			cudaFree(mParameterInfoOnGPU);
		}

	}
}