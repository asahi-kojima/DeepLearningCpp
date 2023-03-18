#include <random>
#include <cuda_runtime.h>
#include <cassert>

#include "Affine.h"
#include "../../../commonGPU.cuh"

namespace Aoba {
	namespace layer
	{
		namespace
		{
			using flowDataType = BaseLayer::flowDataType;


			__global__ void AffineForward(
				flowDataType* y, flowDataType* A, 
				flowDataType* x, flowDataType* b, u32 outputSize, u32 inputSize, u32 batchSize)
			{
				u32 xid = blockIdx.x * blockDim.x + threadIdx.x;
				u32 yid = blockIdx.y * blockDim.y + threadIdx.y;
				if (xid >= outputSize || yid >= batchSize)
				{
					return;
				}
				u32 id = yid * outputSize + xid;

				f32 result = 0.0f;
				for (u32 i = 0; i < inputSize; i++)
				{
#if _DEBUG
					u32 tmp = xid * inputSize + i;
					if (tmp < 0 || tmp >= inputSize * outputSize)
					{
						printf("Affine A parameter : out of range : %d\n", tmp);
						printf("threadId x = %d  ,  y = %d\n", threadIdx.x, threadIdx.y);
						assert(0);
					}
					tmp = yid * inputSize + i;
					if (tmp < 0 || tmp >= inputSize * batchSize)
					{
						printf("Affine x parameter : out of range : %d", tmp);
						assert(0);
					}
#endif
					result += A[xid * inputSize + i] * x[yid * inputSize + i]; 
				}
#if _DEBUG
				if (!(id >= 0 && id < batchSize * outputSize))
				{
					printf("Affine y parameter : out of range : %d", id);
					assert(0);
				}
#endif
				y[id] = result + b[xid];
			}

			__global__ void AffineBackward(flowDataType* dA, flowDataType* dout, flowDataType* input, u32 outputSize, u32 inputSize, u32 batchSize)
			{
				u32 xid = blockIdx.x * blockDim.x + threadIdx.x;
				u32 yid = blockIdx.y * blockDim.y + threadIdx.y;
				if (xid >= inputSize || yid >= outputSize)
				{
					return;
				}

				u32 id = yid * inputSize + xid;

				f32 result = 0.0f;
				for (u32 N = 0; N < batchSize; N++)
				{
#if _DEBUG
					if (N * inputSize + xid >= batchSize * inputSize)
					{
						assert(0);
					}
					if (N * outputSize + yid >= batchSize * outputSize)
					{
						assert(0);
					}
#endif
					result += input[N * inputSize + xid] * dout[N * outputSize + yid];
				}

				dA[id] = result;
			}

			__global__ void biasBackward(flowDataType * dBias, flowDataType * dout, u32 outputSize,u32 batchSize)
			{
				u32 id = blockIdx.x * blockDim.x + threadIdx.x;
				if (id >= outputSize)
				{
					return;
				}
				f32 result = 0.0f;
				for (u32 N = 0; N < batchSize; N++)
				{
#if _DEBUG
					if ((N * outputSize + id) >= batchSize * outputSize)
					{
						assert(0);
					}
#endif
					result += dout[N * outputSize + id];
				}
#if _DEBUG
				if (id >= outputSize)
				{
					assert(0);
				}
#endif
				dBias[id] = result;
			}

			__global__ void doutBackward(flowDataType* dOut ,flowDataType * A, flowDataType* dIn,u32 outputSize, u32 inputSize, u32 batchSize)
			{
				u32 xid = blockIdx.x * blockDim.x + threadIdx.x;//input
				u32 yid = blockIdx.y * blockDim.y + threadIdx.y;//batch
				
				if (xid >= inputSize || yid >= batchSize)
				{
					return;
				}

				f32 result = 0.0f;
				for (u32 i = 0; i < outputSize; i++)
				{
#if _DEBUG
					if (i * inputSize + xid >= outputSize * inputSize)
					{
						assert(0);
					}
					if (yid * outputSize + i >= batchSize * outputSize)
					{
						assert(0);
					}
#endif
					result += A[i * inputSize + xid] * dIn[yid * outputSize + i];
				}
				dOut[yid * inputSize + xid] = result;
			}
		}
		void Affine::initializeOnGPU()
		{
			pParametersOnGPU.resize(2);
			pDParametersOnGPU.resize(2);

			//AffineÉpÉâÉÅÅ[É^
			paramMemory& affineParam = pParametersOnGPU[0];
			paramMemory& affineDParam = pDParametersOnGPU[0];

			affineParam.size = affineDParam.size = mOutputSize * mInputSize;

			CHECK(cudaMalloc((void**)(&(affineParam.address)), affineParam.size * sizeof(parameterType))   );
			CHECK(cudaMalloc((void**)(&(affineDParam.address)), affineDParam.size * sizeof(parameterType)) );

			parameterType* tmpAffineParam = new parameterType[affineParam.size];
			{
				std::random_device seed_gen;
				std::default_random_engine engine(seed_gen());
				std::normal_distribution<> dist(0.0, std::sqrt(2.0 / mInputSize));

				parameterType* tmp = new parameterType[affineParam.size];
				for (u32 idx = 0; idx < affineParam.size; idx++)
				{
					tmp[idx] = mAffineParamWeight * static_cast<f32>(dist(engine)) / std::sqrt(2.0f / mInputSize);
				}
				CHECK(cudaMemcpy(affineParam.address, tmp, affineParam.size * sizeof(parameterType), cudaMemcpyHostToDevice));

				for (u32 idx = 0; idx < affineDParam.size; idx++)
				{
					tmp[idx] = 0.0f;
				}
				CHECK(cudaMemcpy(affineDParam.address, tmp, affineDParam.size * sizeof(parameterType), cudaMemcpyHostToDevice));
				delete[] tmp;
			}


			//BiasÉpÉâÉÅÅ[É^
			paramMemory& biasParam = pParametersOnGPU[1];
			paramMemory& biasDParam = pDParametersOnGPU[1];

			biasParam.size = biasDParam.size = mOutputSize;

			cudaMalloc((void**)(&(biasParam.address)), biasParam.size * sizeof(parameterType));
			cudaMalloc((void**)(&(biasDParam.address)), biasDParam.size * sizeof(parameterType));
			{
				parameterType* tmp = new parameterType[biasParam.size];
				for (u32 idx = 0; idx < biasParam.size; idx++)
				{
					tmp[idx] = 0.0f;
				}
				CHECK(cudaMemcpy(biasParam.address, tmp, biasParam.size * sizeof(parameterType), cudaMemcpyHostToDevice));
				CHECK(cudaMemcpy(biasDParam.address, tmp, biasDParam.size * sizeof(parameterType), cudaMemcpyHostToDevice));
				delete[] tmp;
			}

			//åvéZåãâ Çäiî[Ç∑ÇÈÇΩÇﬂÇÃÉÅÉÇÉäämï€
			mForwardResultOnGPU.size = mBatchSize * mOutputSize;
			mBackwardResultOnGPU.size = mBatchSize * mInputSize;
			CHECK(cudaMalloc((void**)(&(mForwardResultOnGPU.address)), 
				mForwardResultOnGPU.size * sizeof(flowDataType)));
			CHECK(cudaMalloc((void**)(&(mBackwardResultOnGPU.address)), 
				mBackwardResultOnGPU.size * sizeof(flowDataType)));
			{
				flowDataType* tmp = new flowDataType[mForwardResultOnGPU.size];
				for (u32 idx = 0; idx < mForwardResultOnGPU.size; idx++)
				{
					tmp[idx] = 0.0f;
				}
				CHECK(cudaMemcpy(mForwardResultOnGPU.address, tmp, 
					mForwardResultOnGPU.size * sizeof(flowDataType), cudaMemcpyHostToDevice));
				delete[] tmp;


				tmp = new flowDataType[mBackwardResultOnGPU.size];
				for (u32 idx = 0; idx < mBackwardResultOnGPU.size; idx++)
				{
					tmp[idx] = 0.0f;
				}
				CHECK(cudaMemcpy(mBackwardResultOnGPU.address, tmp, 
					mBackwardResultOnGPU.size * sizeof(flowDataType), cudaMemcpyHostToDevice));
				delete[] tmp;
			}
		}

		void Affine::forwardOnGPU()
		{
			dim3 block(16,16);
			dim3 grid(
				(mOutputSize + block.x - 1) / block.x,
				(mBatchSize + block.y - 1) / block.y);

			AffineForward << <grid, block >> > (
				mForwardResultOnGPU.address,
				pParametersOnGPU[0].address,
				mInputDataOnGPU->address,
				pParametersOnGPU[1].address,
				mOutputSize,
				mInputSize,
				mBatchSize);
#if _DEBUG
			CHECK(cudaDeviceSynchronize());
#endif
		}

		void Affine::backwardOnGPU()
		{
			//doutÇÃãtì`î¿
			{
				dim3 block(16, 16);
				dim3 grid(
					(mInputSize + block.x - 1) / block.x,
					(mBatchSize + block.y - 1) / block.y);
				doutBackward << <grid, block >> > (
					mBackwardResultOnGPU.address,
					pParametersOnGPU[0].address,
					mDInputDataOnGPU->address,
					mOutputSize,
					mInputSize,
					mBatchSize);
#if _DEBUG
				CHECK(cudaDeviceSynchronize());
#endif
			}

			//AÇÃãtì`î¿
			{
				dim3 block(16,16);
				dim3 grid(
					(mOutputSize + block.x - 1) / block.x,
					(mInputSize + block.y - 1) / block.y);
				AffineBackward << <grid, block >> > (
					pDParametersOnGPU[0].address,
					mDInputDataOnGPU->address,
					mInputDataOnGPU->address,
					mOutputSize,
					mInputSize,
					mBatchSize);
#if _DEBUG
				CHECK(cudaDeviceSynchronize());
#endif
			}

			//BiasÇÃãtì`î¿
			{
				dim3 block(16);
				dim3 grid((mOutputSize + block.x - 1) / block.x);
				biasBackward << <grid, block >> > (
					pDParametersOnGPU[1].address,
					mDInputDataOnGPU->address,
					mOutputSize,
					mBatchSize);
#if _DEBUG
				CHECK(cudaDeviceSynchronize());
#endif
			}
		}

		void Affine::terminateOnGPU()
		{

		}

	}
}