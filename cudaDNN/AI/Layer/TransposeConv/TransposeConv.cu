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
#include "TransposeConv.h"


namespace Aoba {
	namespace layer
	{
		namespace
		{


			__global__ void cudaCreateReshapedData(
				f32* reshapedData, f32* input,
				TransposeConv::parameterInfo* pInfo, u32 batchSize, u32 dataSize)
			{
				TransposeConv::parameterInfo info = *pInfo;

				const u32 N = blockIdx.x * blockDim.x + threadIdx.x;
				const u32 id = blockIdx.y * blockDim.y + threadIdx.y;

				if (N >= batchSize || id >= dataSize)
				{
					return;
				}

				const u32 mIhIw = info.IhIw;
				const u32 mIw = info.Iw;

				const u32 Ic = id / mIhIw;
				const u32 Ih = (id - Ic * mIhIw) / mIw;
				const u32 Iw = id % mIw;


				const u32 mSh = info.Sh;
				const u32 mSw = info.Sw;

				const u32 mFh = info.Fh;
				const u32 mFw = info.Fw;
				const u32 mFhFw = mFh * mFw;

				const u32 mOh = info.Oh;
				const u32 mOw = info.Ow;

				const u32 mOhOwIcFhFw = info.OhOwIcFhFw;
				const u32 mIcFhFw = info.IcFhFw;

				const u32 exH = mFh - 1 - info.Ph + Ih * mSh;
				const u32 exW = mFw - 1 - info.Pw + Iw * mSw;

				f32 value = input[N * dataSize + id];

				for (u32 Oh = (exH < mFh ? 0 : 1 + (exH - mFh) / 1), endOh = min(1 + (exH / 1), mOh); Oh < endOh; Oh++)
				{
					for (u32 Ow = (exW < mFw ? 0 : 1 + (exW - mFw) / 1), endOw = min(1 + (exW / 1), mOw); Ow < endOw; Ow++)
					{
						const u32 row = Oh * mOw + Ow;
						const u32 col = Ic * mFhFw + (exH - Oh * 1) * mFw + (exW - Ow * 1);
						reshapedData[N * mOhOwIcFhFw + row * mIcFhFw + col] = value;
					}
				}
			}

			__global__ void cudaTransposeConvForward(
				f32* y, f32* FMatrix,
				f32* reshapedInput, f32* bias, TransposeConv::parameterInfo* pInfo, u32 batchSize)
			{
				TransposeConv::parameterInfo info = *pInfo;


				const u32 N = blockIdx.x * blockDim.x + threadIdx.x;
				const u32 OcOhOw = blockIdx.y * blockDim.y + threadIdx.y;

				const u32 mOcOhOw = info.OcOhOw;
				if (N >= batchSize || OcOhOw >= mOcOhOw)
				{
					return;
				}

				const u32 id = N * mOcOhOw + OcOhOw;

				const u32 mOhOw = info.OhOw;
				const u32 mFc = OcOhOw / mOhOw;
				const u32 OhOw = OcOhOw - mFc * mOhOw;

				const u32 mIcFhFw = info.IcFhFw;
				const u32 mOhOwIcFhFw = info.OhOwIcFhFw;

				f32 result = 0.0f;
				for (u32 i = 0; i < mIcFhFw; i++)
				{

					result += FMatrix[mFc * mIcFhFw + i] * reshapedInput[N * mOhOwIcFhFw + OhOw * mIcFhFw + i];
				}

				y[id] = result + bias[mFc];
			}




			__global__ void backwardOnGPU_filter(f32* dFilter, f32* dout, f32* reshapedInput,
				TransposeConv::parameterInfo* pInfo, u32 batchSize)
			{
				TransposeConv::parameterInfo info = *pInfo;

				u32 c = blockIdx.x * blockDim.x + threadIdx.x;
				u32 icfhfw = blockIdx.y * blockDim.y + threadIdx.y;
				if (c >= info.Oc || icfhfw >= info.IcFhFw)
				{
					return;
				}


				const u32 mOhOw = info.OhOw;
				const u32 mOcOhOw = info.OcOhOw;
				const u32 mOhOwIcFhFw = info.OhOwIcFhFw;
				const u32 mIcFhFw = info.IcFhFw;

				f32 result = 0.0f;
				for (u32 N = 0; N < batchSize; N++)
				{
					for (u32 hw = 0; hw < mOhOw; hw++)
					{
						result += dout[N * mOcOhOw + c * mOhOw + hw] * reshapedInput[N * mOhOwIcFhFw + hw * mIcFhFw + icfhfw];
					}
				}
				dFilter[c * mIcFhFw + icfhfw] = result;
			}

			__global__ void backwardOnGPU_bias(f32* dBias, f32* dout, TransposeConv::parameterInfo* pInfo, u32 batchSize)
			{
				TransposeConv::parameterInfo info = *pInfo;

				u32 id = blockIdx.x * blockDim.x + threadIdx.x;
				if (id >= info.Oc)
				{
					return;
				}

				const u32 OcOhOw = info.OcOhOw;
				const u32 OhOw = info.OhOw;

				f32 result = 0.0f;
				for (u32 N = 0; N < batchSize; N++)
				{
					for (u32 hw = 0; hw < OhOw; hw++)
					{
						result += dout[N * OcOhOw + id * OhOw + hw];
					}
				}

				dBias[id] = result;
			}

			__global__ void backwardOnGPU_dout(
				f32* backwardResult,
				f32* dOut,
				f32* FMatrix,
				TransposeConv::parameterInfo* pInfo,
				u32 batchSize)
			{
				TransposeConv::parameterInfo info = *pInfo;

				u32 N = blockIdx.x * blockDim.x + threadIdx.x;//input
				u32 IcIhIw = blockIdx.y * blockDim.y + threadIdx.y;//batch

				const u32 mIcIhIw = info.IcIhIw;
				if (N >= batchSize || IcIhIw >= mIcIhIw)
				{
					return;
				}

				u32 id = N * mIcIhIw + IcIhIw;

				const u32 mIhIw = info.IhIw;
				const u32 mIw = info.Iw;

				const u32 c = IcIhIw / mIhIw;
				const u32 h = (IcIhIw - c * mIhIw) / mIw;
				const u32 w = IcIhIw % mIw;

				const u32 mOh = info.Oh;
				const u32 mOw = info.Ow;
				const u32 mFh = info.Fh;
				const u32 mFw = info.Fw;
				const u32 mFhFw = info.FhFw;
				const u32 mFn = info.Fn;
				const u32 mSh = info.Sh;
				const u32 mSw = info.Sw;

				const u32 mOcOhOw = info.OcOhOw;
				const u32 mOhOw = info.OhOw;
				const u32 mIcFhFw = info.IcFhFw;
				
				const u32 exH = mFh - 1 - info.Ph + h * mSh;
				const u32 exW = mFw - 1 - info.Pw + w * mSw;

				f32 result = 0.0f;
				for (u32 oh = (exH < mFh ? 0 : 1 + (exH - mFh) / 1), endOh = min(1 + (exH / 1), mOh); oh < endOh; oh++)
				{
					for (u32 ow = (exW < mFw ? 0 : 1 + (exW - mFw) / 1), endOw = min(1 + (exW / 1), mOw); ow < endOw; ow++)
					{
						const u32 row = oh * mOw + ow;
						const u32 col = c * mFhFw + (exH - oh * 1) * mFw + (exW - ow * 1);
						for (u32 Fc = 0; Fc < mFn; Fc++)
						{
							result += dOut[N * mOcOhOw + Fc * mOhOw + row] * FMatrix[Fc * mIcFhFw + col];
						}
					}
				}
				backwardResult[id] = result;
			}
		}
		void TransposeConv::mallocOnGPU()
		{
			mParametersPtrOnGPU.resize(2);
			mDParametersPtrOnGPU.resize(2);

			//Affineパラメータ
			DataArray& convParam = mParametersPtrOnGPU[0];
			DataArray& convDParam = mDParametersPtrOnGPU[0];

			convParam.size = convDParam.size = mOc * mIcFhFw;

			MALLOC_AND_INITIALIZE_NORMAL_ON_GPU(convParam, mIcFhFw, mTransposeConvParamWeight);
			MALLOC_AND_INITIALIZE_0_ON_GPU(convDParam);


			//Biasパラメータ 
			DataArray& biasParam = mParametersPtrOnGPU[1];
			DataArray& biasDParam = mDParametersPtrOnGPU[1];

			biasParam.size = biasDParam.size = mOutputDataShape.channel;

			MALLOC_AND_INITIALIZE_0_ON_GPU(biasParam);
			MALLOC_AND_INITIALIZE_0_ON_GPU(biasDParam);



			//計算結果を格納するためのメモリ確保
			mForwardResultOnGPU.setSizeAs4D(mBatchSize, mOc, mOh, mOw);
			mReshapedInputDataOnGPU.setSizeAs3D(mBatchSize, mOhOw, mIcFhFw);
			mBackwardResultOnGPU.setSizeAs4D(mBatchSize, mIc, mIh, mIw);

			MALLOC_AND_INITIALIZE_0_ON_GPU(mForwardResultOnGPU);
			MALLOC_AND_INITIALIZE_0_ON_GPU(mReshapedInputDataOnGPU);
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
			tmp.OhOwIcFhFw = mOhOw * mIcFhFw;
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

		void TransposeConv::forwardOnGPU()
		{
			auto& input = *mInputDataOnGPU;
			DataArray& convMatrix = mParametersPtrOnGPU[0];
			DataArray& convBias = mParametersPtrOnGPU[1];

			{
				dim3 block(16, 16);
				dim3 grid(
					(mBatchSize + block.x - 1) / block.x,
					(mIcIhIw + block.y - 1) / block.y);
#if TIME_DEBUG
				{
					std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
					cudaCreateReshapedData << <grid, block >> >
						(mReshapedInputDataOnGPU.address,
							input.address,
							mParameterInfoOnGPU,
							mBatchSize,
							mIcIhIw);

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

			{
				dim3 block(16, 32);
				dim3 grid(
					(mBatchSize + block.x - 1) / block.x,
					(mOcOhOw + block.y - 1) / block.y);
#if TIME_DEBUG
				{
					std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
					cudaTransposeConvForward << <grid, block >> >
						(mForwardResultOnGPU.address,
							convMatrix.address,
							mReshapedInputDataOnGPU.address,
							convBias.address,
							mParameterInfoOnGPU,
							mBatchSize);

#if GPU_SYNC_DEBUG
					CHECK(cudaDeviceSynchronize());
#endif
#if TIME_DEBUG
					f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
					std::string name = "";
					(((name += __FUNCTION__) += " : ") += std::to_string(mInstanceID)) += " : cudaTransposeConvForward";
					timers[name] = elapsedTime;
				}
#endif
			}
		}

		void TransposeConv::backwardOnGPU()
		{
			auto& dout = *mDInputDataOnGPU;

			DataArray& convMatrix = mParametersPtrOnGPU[0];
			DataArray& dConvMatrix = mDParametersPtrOnGPU[0];
			DataArray& convBias = mParametersPtrOnGPU[1];
			DataArray& dConvBias = mDParametersPtrOnGPU[1];


			{
				dim3 block(16, 16);
				dim3 grid(
					(mOc + block.x - 1) / block.x,
					(mIcFhFw + block.y - 1) / block.y);
#if TIME_DEBUG
				{
					std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
					backwardOnGPU_filter << <grid, block >> > (
						dConvMatrix.address,
						dout.address,
						mReshapedInputDataOnGPU.address,
						mParameterInfoOnGPU,
						mBatchSize);

#if GPU_SYNC_DEBUG
					CHECK(cudaDeviceSynchronize());
#endif
#if TIME_DEBUG
					f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
					std::string name = "";
					(((name += __FUNCTION__) += " : ") += std::to_string(mInstanceID)) += " : backwardOnGPU_filter";
					timers[name] = elapsedTime;
				}
#endif
			}

			{
				dim3 block(16);
				dim3 grid((mOc + block.x - 1) / block.x);
#if TIME_DEBUG
				{
					std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
					backwardOnGPU_bias << <grid, block >> > (dConvBias.address, dout.address, mParameterInfoOnGPU, mBatchSize);

#if GPU_SYNC_DEBUG
					CHECK(cudaDeviceSynchronize());
#endif
#if TIME_DEBUG
					f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
					std::string name = "";
					(((name += __FUNCTION__) += " : ") += std::to_string(mInstanceID)) += " : backwardOnGPU_bias";
					timers[name] = elapsedTime;
				}
#endif
			}

			{
				dim3 block(16, 16);
				dim3 grid(
					(mBatchSize + block.x - 1) / block.x,
					(mIcIhIw + block.y - 1) / block.y);
#if TIME_DEBUG
				{
					std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
					backwardOnGPU_dout << <grid, block >> > (
						mBackwardResultOnGPU.address,
						dout.address, convMatrix.address,
						mParameterInfoOnGPU,
						mBatchSize);

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

		void TransposeConv::terminateOnGPU()
		{
			for (u32 id = 0; id < mParametersPtrOnCPU.size(); id++)
			{
				CUDA_FREE(mParametersPtrOnGPU[id]);
				CUDA_FREE(mDParametersPtrOnGPU[id]);
			}

			CUDA_FREE(mForwardResultOnGPU);
			CUDA_FREE(mReshapedInputDataOnGPU);
			CUDA_FREE(mBackwardResultOnGPU);

			cudaFree(mParameterInfoOnGPU);
		}

	}
}