#include <random>
#include <cuda_runtime.h>
#include <cassert>
#include <device_functions.h>

//このマクロはCUDAファイルがコンパイルされる時に定義される。
//インテリセンスのエラーを一時的に抑制するためにこの定義を置いている。
#if !defined(__CUDACC__)
#define __CUDACC__
#endif



#include "Convolution.h"
#include "../../AIMacro.h"
#include "../../../common.h"
#include "../../../setting.h"


namespace Aoba {
	namespace layer
	{
		namespace
		{


			__global__ void cudaCreateReshapedData(
				f32* reshapedData, f32* input,
				Convolution::parameterInfo* pInfo, u32 batchSize, u32 reshapedDataSize)
			{
				Convolution::parameterInfo info = *pInfo;

				const u32 N = blockIdx.x * blockDim.x + threadIdx.x;
				const u32 id = blockIdx.y * blockDim.y + threadIdx.y;

				if (N >= batchSize || id >= reshapedDataSize)
				{
					return;
				}

				const u32 IcFhFw = info.IcFhFw;
				const u32 V = id / IcFhFw;
				const u32 H = id - V * IcFhFw;
				 
				const u32 Ow = info.Ow;
				const u32 Fh = V / Ow;
				const u32 Fw = V - Fh * Ow;
				
				const u32 FhFw = info.FhFw;
				const u32 filterW = info.Fw;
				const u32 c = H / FhFw;
				const u32 h = (H - c * FhFw) / filterW;
				const u32 w = H % filterW;
				
				const u32 indexH = Fh * info.Sh + (h - info.Ph);
				const u32 indexW = Fw * info.Sw + (w - info.Pw);

				bool isValid = !(indexH < 0 || indexH >= info.Ih ||
					indexW < 0 || indexW >= info.Iw);
				reshapedData[id] = isValid ?
					input[N * info.IcIhIw + c * info.IhIw + indexH * info.Iw + indexW] : 0;
			}

			__global__ void cudaConvolutionForward(
				f32* y, f32* FMatrix,
				f32* input, f32* bias, Convolution::parameterInfo* pInfo, u32 batchSize, u32 OcOhOwSize)
			{
				Convolution::parameterInfo info = *pInfo;


				const u32 N = blockIdx.x * blockDim.x + threadIdx.x;
				const u32 OcOhOw = blockIdx.y * blockDim.y + threadIdx.y;
				if (N >= batchSize || OcOhOw >= OcOhOwSize)
				{
					return;
				}

				const u32 id = N * OcOhOwSize + OcOhOw;

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

			//__global__ void backwardOnGPU_dout_init(f32* backwardResult, u32 batchSize, u32 IcIhIwRange)
			//{
			//	u32 N = blockIdx.x * blockDim.x + threadIdx.x;
			//	u32 IcIhIw = blockIdx.y * blockDim.y + threadIdx.y;

			//	if (N >= batchSize || IcIhIw >= IcIhIwRange)
			//	{
			//		return;
			//	}

			//	backwardResult[N * IcIhIwRange + IcIhIw] = 0.0;
			//}


			__global__ void backwardOnGPU_dout(f32* dFilter, f32* dout, f32* reshapedInput, Convolution::parameterInfo* pInfo)
			{}

			__global__ void backwardOnGPU_filter(f32* dFilter, f32* dout, f32* reshapedInput,
				u32 channelSize, u32 icfhfwSize, u32 batchSize, u32 OhOw, u32 IcFhFw)
			{
				u32 c = blockIdx.x * blockDim.x + threadIdx.x;
				u32 icfhfw = blockIdx.y * blockDim.y + threadIdx.y;
				if (c >= channelSize || icfhfw >= icfhfwSize)
				{
					return;
				}


				f32 result = 0.0f;
				for (u32 N = 0; N < batchSize; N++)
				{
					for (u32 hw = 0; hw < OhOw; hw++)
					{
						result += dout[N * (channelSize * OhOw) + c * OhOw + hw] * reshapedInput[N * (OhOw * IcFhFw) + hw * IcFhFw + icfhfw];
					}
				}

				dFilter[c * IcFhFw + icfhfw] = result;
			}

			__global__ void backwardOnGPU_bias(f32* dBias, f32* dout, u32 dBiasSize, u32 batchSize, u32 OhOw)
			{
				u32 id = blockIdx.x * blockDim.x + threadIdx.x;
				if (id >= dBiasSize)
				{
					return;
				}

				f32 result = 0.0f;
				for (u32 N = 0; N < batchSize; N++)
				{
					for (u32 hw = 0; hw < OhOw; hw++)
					{
						result += dout[N * (dBiasSize * OhOw) + id * OhOw + hw];
					}
				}

				dBias[id] = result;
			}
			//
			//			__global__ void doutBackward(f32* dBias, f32* A, f32* dIn, u32 outputSize, u32 inputSize, u32 batchSize)
			//			{
			//				u32 xid = blockIdx.x * blockDim.x + threadIdx.x;//input
			//				u32 yid = blockIdx.y * blockDim.y + threadIdx.y;//batch
			//
			//				if (xid >= inputSize || yid >= batchSize)
			//				{
			//					return;
			//				}
			//
			//				f32 result = 0.0f;
			//				for (u32 i = 0; i < outputSize; i++)
			//				{
			//#if _DEBUG
			//					if (i * inputSize + xid >= outputSize * inputSize)
			//					{
			//						assert(0);
			//					}
			//					if (yid * outputSize + i >= batchSize * outputSize)
			//					{
			//						assert(0);
			//					}
			//#endif
			//					result += A[i * inputSize + xid] * dIn[yid * outputSize + i];
			//				}
			//				dOut[yid * inputSize + xid] = result;
			//				//printf("dOut[%d * %d + %d] = %lf\n",yid, inputSize, xid, dOut[yid * inputSize + xid]);
			//			}
		}
		void Convolution::mallocOnGPU()
		{
			mParametersPtrOnGPU.resize(2);
			mDParametersPtrOnGPU.resize(2);

			//Affineパラメータ
			DataArray& convParam = mParametersPtrOnGPU[0];
			DataArray& convDParam = mDParametersPtrOnGPU[0];

			convParam.size = convDParam.size = mFilterNum * mIcFhFw;


			MALLOC_AND_INITIALIZE_NORMAL_ON_GPU(convParam, 1, mConvolutionParamWeight);
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




			//
			parameterInfo tmp;
			tmp.batchSize = mBatchSize;
			tmp.Ih = mIh;
			tmp.Iw = mIw;
			tmp.Ic = mIc;
			tmp.IcIhIw = mIcIhIw;
			tmp.IcFhFw = mIcFhFw;
			tmp.Ow = mOw;
			tmp.OhOw = mOhOw;
			tmp.FhFw = mFhFw;
			tmp.OhOwIcFhFw = mOhOw * mIcFhFw;
			tmp.IhIw = mIh * mIw;
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
				dim3 block(16, 32);
				dim3 grid(
					(mBatchSize + block.x - 1) / block.x,
					(mIcFhFw * mOhOw + block.y - 1) / block.y);
				cudaCreateReshapedData << <grid, block >> >
					(mReshapedInputDataOnGPU.address,
						input.address,
						mParameterInfoOnGPU,
						mBatchSize,
						mIcFhFw * mOhOw);

#if GPU_SYNC_DEBUG
				CHECK(cudaDeviceSynchronize());
#endif
			}

			{
				dim3 block(16, 32);
				dim3 grid(
					(mBatchSize + block.x - 1) / block.x,
					(mOcOhOw + block.y - 1) / block.y);
				cudaConvolutionForward << <grid, block >> >
					(mForwardResultOnGPU.address,
						convMatrix.address,
						mReshapedInputDataOnGPU.address,
						convBias.address,
						mParameterInfoOnGPU,
						mBatchSize,
						mOcOhOw);

#if GPU_SYNC_DEBUG
				CHECK(cudaDeviceSynchronize());
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
				backwardOnGPU_filter << <grid, block >> >
					(dConvMatrix.address, dout.address, mReshapedInputDataOnGPU.address,
						mOc, mIcFhFw, mBatchSize, mOhOw, mIcFhFw);

#if GPU_SYNC_DEBUG
				CHECK(cudaDeviceSynchronize());
#endif
			}

			{
				dim3 block(16);
				dim3 grid((mOc + block.x - 1) / block.x);

				backwardOnGPU_bias << <grid, block >> > (dConvBias.address, dout.address, mOc, mBatchSize, mOhOw);

#if GPU_SYNC_DEBUG
				CHECK(cudaDeviceSynchronize());
#endif
			}

			{
				dim3 block(16, 16);
				dim3 grid(
					(mBatchSize + block.x - 1) / block.x,
					(mIcFhFw + block.y - 1) / block.y);

				//backwardOnGPU_dout << <grid, block >> > (dConvBias.address, dout.address, mOc, mBatchSize, mOhOw);

#if GPU_SYNC_DEBUG
				CHECK(cudaDeviceSynchronize());
#endif
			}
		}

		void Convolution::terminateOnGPU()
		{

		}

	}
}