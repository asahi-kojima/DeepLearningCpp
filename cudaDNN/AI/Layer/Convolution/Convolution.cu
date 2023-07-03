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
#include "Convolution.h"


namespace Aoba {
	namespace layer
	{
		namespace
		{


			__global__ void cudaCreateReshapedData(
				f32* reshapedData, f32* input,
				Convolution::parameterInfo* pInfo, u32 batchSize, u32 dataSize)
			{
				Convolution::parameterInfo info = *pInfo;

				const u32 N = blockIdx.x * blockDim.x + threadIdx.x;
				const u32 id = blockIdx.y * blockDim.y + threadIdx.y;

				if (N >= batchSize || id >= dataSize)
				{
					return;
				}

				const u32 Ic = id / info.IhIw;
				const u32 Ih = (id - Ic * info.IhIw) / info.Iw;
				const u32 Iw = id % info.Iw;

				const u32 exH = Ih + info.Ph;
				const u32 exW = Iw + info.Pw;

				f32 value = input[N * dataSize + id];

				for (u32 Oh = (exH < info.Fh ? 0 : 1 + (exH - info.Fh) / info.Sh), endOh = min(1 + (exH / info.Sh), info.Oh); Oh < endOh; Oh++)
				{
					for (u32 Ow = (exW < info.Fw ? 0 : 1 + (exW - info.Fw) / info.Sw), endOw = min(1 + (exW / info.Sw), info.Ow); Ow < endOw; Ow++)
					{
						const u32 row = Oh * info.Ow + Ow;
						const u32 col = Ic * info.FhFw + (exH - Oh * info.Sh) * info.Fw + (exW - Ow * info.Sw);
						reshapedData[N * info.OhOwIcFhFw + row * info.IcFhFw + col] = value;
					}
				}
			}

			__global__ void cudaConvolutionForward(
				f32* y, f32* FMatrix,
				f32* input, f32* bias, Convolution::parameterInfo* pInfo, u32 batchSize)
			{
				Convolution::parameterInfo info = *pInfo;


				const u32 N = blockIdx.x * blockDim.x + threadIdx.x;
				const u32 OcOhOw = blockIdx.y * blockDim.y + threadIdx.y;
				if (N >= batchSize || OcOhOw >= info.OcOhOw)
				{
					return;
				}

				const u32 id = N * info.OcOhOw + OcOhOw;

				const u32 Fc = OcOhOw / info.OhOw;
				const u32 OhOw = OcOhOw - Fc * info.OhOw;

				const u32 IcFhFw = info.IcFhFw;
				const u32 OhOwIcFhFw = info.OhOwIcFhFw;

				f32 result = 0.0f;
				for (u32 i = 0, end = info.IcFhFw; i < end; i++)
				{

					result += FMatrix[Fc * IcFhFw + i] * input[N * OhOwIcFhFw + OhOw * IcFhFw + i];
				}

				y[id] = result + bias[Fc];
			}




			__global__ void backwardOnGPU_filter(f32* dFilter, f32* dout, f32* reshapedInput,
				Convolution::parameterInfo* pInfo, u32 batchSize)
			{
				Convolution::parameterInfo info = *pInfo;

				u32 c = blockIdx.x * blockDim.x + threadIdx.x;
				u32 icfhfw = blockIdx.y * blockDim.y + threadIdx.y;
				if (c >= info.Oc || icfhfw >= info.IcFhFw)
				{
					return;
				}


				const u32 OhOw = info.OhOw;
				const u32 OcOhOw = info.OcOhOw;
				const u32 OhOwIcFhFw = info.OhOwIcFhFw;
				const u32 IcFhFw = info.IcFhFw;

				f32 result = 0.0f;
				for (u32 N = 0; N < batchSize; N++)
				{
					for (u32 hw = 0; hw < OhOw; hw++)
					{
						result += dout[N * OcOhOw + c * OhOw + hw] * reshapedInput[N * OhOwIcFhFw + hw * IcFhFw + IcFhFw];
					}
				}
				dFilter[c * IcFhFw + icfhfw] = result;
			}

			__global__ void backwardOnGPU_bias(f32* dBias, f32* dout, Convolution::parameterInfo* pInfo, u32 batchSize)
			{
				Convolution::parameterInfo info = *pInfo;

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
				Convolution::parameterInfo* pInfo,
				u32 batchSize)
			{
				Convolution::parameterInfo info = *pInfo;

				u32 N = blockIdx.x * blockDim.x + threadIdx.x;//input
				u32 IcIhIw = blockIdx.y * blockDim.y + threadIdx.y;//batch
				if (N >= batchSize || IcIhIw >= info.IcIhIw)
				{
					return;
				}

				u32 id = N * info.IcIhIw + IcIhIw;

				const u32 c = IcIhIw / info.IhIw;
				const u32 h = (IcIhIw - c * info.IhIw) / info.Iw;
				const u32 w = IcIhIw % info.Iw;

				const u32 exH = h + info.Ph;
				const u32 exW = w + info.Pw;

				const u32 Oh = info.Oh;
				const u32 Ow = info.Ow;
				const u32 Fh = info.Fh;
				const u32 Fw = info.Fw;
				const u32 FhFw = info.FhFw;
				const u32 Fn = info.Fn;
				const u32 Sh = info.Sh;
				const u32 Sw = info.Sw;

				f32 result = 0.0f;
				for (u32 oh = (exH < Fh ? 0 : 1 + (exH - Fh) / Sh), endOh = min(1 + (exH / Sh), Oh); oh < endOh; oh++)
				{
					for (u32 ow = (exW < Fw ? 0 : 1 + (exW - Fw) / Sw), endOw = min(1 + (exW / Sw), Ow); ow < endOw; ow++)
					{
						const u32 row = oh * Ow + ow;
						const u32 col = c * FhFw + (exH - oh * Sh) * Fw + (exW - ow * Sw);
						for (u32 Fc = 0; Fc < Fn; Fc++)
						{
							result += dOut[N * info.OcOhOw + Fc * info.OhOw + row] * FMatrix[Fc * info.IcFhFw + col];
						}
					}
				}
				backwardResult[id] = result;
			}
		}
		void Convolution::mallocOnGPU()
		{
			mParametersPtrOnGPU.resize(2);
			mDParametersPtrOnGPU.resize(2);

			//Affineパラメータ
			DataArray& convParam = mParametersPtrOnGPU[0];
			DataArray& convDParam = mDParametersPtrOnGPU[0];

			convParam.size = convDParam.size = mFilterNum * mIcFhFw;

			MALLOC_AND_INITIALIZE_NORMAL_ON_GPU(convParam, mIcFhFw, mConvolutionParamWeight);
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
			tmp.Fh = mFilterHeight;
			tmp.Fw = mFilterWidth;
			tmp.Sh = mStrideHeight;
			tmp.Sw = mStrideWidth;
			tmp.Ph = mPaddingHeight;
			tmp.Pw = mPaddingWidth;
			CHECK(cudaMalloc(&mParameterInfoOnGPU, sizeof(parameterInfo)););
			CHECK(cudaMemcpy(mParameterInfoOnGPU, &tmp, sizeof(parameterInfo), cudaMemcpyHostToDevice));
		}

		void Convolution::forwardOnGPU()
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
					cudaConvolutionForward << <grid, block >> >
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
					(((name += __FUNCTION__) += " : ") += std::to_string(mInstanceID)) += " : cudaConvolutionForward";
					timers[name] = elapsedTime;
				}
#endif
			}
		}

		void Convolution::backwardOnGPU()
		{
			auto& dout = *mDInputDataOnGPU;

			DataArray& convMatrix = mParametersPtrOnGPU[0];
			DataArray& dConvMatrix = mDParametersPtrOnGPU[0];
			DataArray& convBias = mParametersPtrOnGPU[1];
			DataArray& dConvBias = mDParametersPtrOnGPU[1];
			//			//doutの逆伝搬
			//			{
			//				dim3 block(16, 16);
			//				dim3 grid(
			//					(mInputSize + block.x - 1) / block.x,
			//					(mBatchSize + block.y - 1) / block.y);
			//				doutBackward << <grid, block >> > (
			//					mBackwardResultOnGPU.address,
			//					mParametersPtrOnGPU[0].address,
			//					mDInputDataOnGPU->address,
			//					mOutputSize,
			//					mInputSize,
			//					mBatchSize);
			//#if _DEBUG
			//				CHECK(cudaDeviceSynchronize());
			//#endif
			//			}
			//
			//			//Aの逆伝搬
			//			{
			//				dim3 block(16, 16);
			//				dim3 grid(
			//					(mInputSize + block.x - 1) / block.x,
			//					(mOutputSize + block.y - 1) / block.y);
			//
			//				AffineBackward << <grid, block >> > (
			//					pDParametersOnGPU[0].address,
			//					mDInputDataOnGPU->address,
			//					mInputDataOnGPU->address,
			//					mOutputSize,
			//					mInputSize,
			//					mBatchSize);
			//
			//#if _DEBUG
			//				CHECK(cudaDeviceSynchronize());
			//#endif
			//			}
			//
			//			//Biasの逆伝搬
			//			{
			//				dim3 block(16);
			//				dim3 grid((mOutputSize + block.x - 1) / block.x);
			//
			//				biasBackward << <grid, block >> > (
			//					pDParametersOnGPU[1].address,
			//					mDInputDataOnGPU->address,
			//					mOutputSize,
			//					mBatchSize);
			//
			//#if _DEBUG
			//				CHECK(cudaDeviceSynchronize());
			//#endif
			//			}

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

		void Convolution::terminateOnGPU()
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