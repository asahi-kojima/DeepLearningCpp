#include <random>
#include <cuda_runtime.h>
#include <cassert>
#include <device_functions.h>
#include <chrono>
//このマクロはCUDAファイルがコンパイルされる時に定義される。
//インテリセンスのエラーを一時的に抑制するためにこの定義を置いている。
#if !defined(__CUDACC__)
#define __CUDACC__
#endif

#include "../../AIHelperFunction.h"
#include "Affine.h"



namespace Aoba {
	namespace layer
	{
		namespace
		{


			__global__ void AffineForward(
				f32* y, f32* A,
				f32* x, f32* b, u32 outputSize, u32 inputSize, u32 batchSize)
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
					//#if _DEBUG
					//					u32 tmp = xid * inputSize + i;
					//					if (tmp < 0 || tmp >= inputSize * outputSize)
					//					{
					//						printf("Affine A parameter : out of range : %d\n", tmp);
					//						printf("threadId x = %d  ,  y = %d\n", threadIdx.x, threadIdx.y);
					//						assert(0);
					//					}
					//					tmp = yid * inputSize + i;
					//					if (tmp < 0 || tmp >= inputSize * batchSize)
					//					{
					//						printf("Affine x parameter : out of range : %d", tmp);
					//						assert(0);
					//					}
					//#endif
					result += A[xid * inputSize + i] * x[yid * inputSize + i];
				}
				//#if _DEBUG
				//				if (!(id >= 0 && id < batchSize * outputSize))
				//				{
				//					printf("Affine y parameter : out of range : %d", id);
				//					assert(0);
				//				}
				//#endif
				y[id] = result + b[xid];
			}


			//struct DataShape
			//{
			//	u32 inputSize;
			//	u32 outputSize;
			//	u32 batchSize;
			//	u32 blockSize;
			//};

			__global__ void AffineForwardWithSM(
				f32* y, f32* A,
				f32* X, f32* b, u32 outputSize, u32 inputSize, u32 batchSize)
			{
				u32 outputID = blockIdx.x * blockDim.x + threadIdx.x;
				u32 batchID = blockIdx.y * blockDim.y + threadIdx.y;


				const u32 BlockSize = blockDim.x;
				const u32 subMatSize = BlockSize * BlockSize;

				const u32 matASize = inputSize * outputSize;
				const u32 matXSize = inputSize * batchSize;
				const u32 subMatAXSize = inputSize * BlockSize;
				const u32 startPointOfA = inputSize * (blockIdx.x * BlockSize);
				const u32 startPointOfX = inputSize * (blockIdx.y * BlockSize);

				extern __shared__ f32 shareRegion[];
				f32* subA = shareRegion;
				f32* subX = shareRegion + inputSize * BlockSize;

				const u32 groupID = (threadIdx.y * blockDim.x + threadIdx.x);

				for (u32 i = 0, loopSize = (inputSize * BlockSize + subMatSize - 1) / subMatSize; i < loopSize; i++)
				{
					u32 index = i + groupID * loopSize;
					if (index >= subMatAXSize)
					{
						continue;
					}

					u32 indexA = startPointOfA + index;
					if (indexA < matASize)
					{
						subA[index] = A[indexA];
					}

					u32 indexX = startPointOfX + index;
					if (indexX < matXSize)
					{
						subX[index] = X[indexX];
					}
				}
				__syncthreads();




				if (outputID >= outputSize || batchID >= batchSize)
				{
					return;
				}




				f32 result = 0.0f;
				for (int i = 0; i < inputSize; i++)
				{
					result += subA[threadIdx.x * inputSize + i] * subX[threadIdx.y * inputSize + i];
				}

				y[batchID * outputSize + outputID] = result + b[outputID];
			}

			__global__ void AffineBackward(f32* dA, f32* dout, f32* input, u32 outputSize, u32 inputSize, u32 batchSize)
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
#if INDEX_DEBUG
					if (N * inputSize + xid >= batchSize * inputSize)
					{
						assert(0);
					}
					if (N * outputSize + yid >= batchSize * outputSize)
					{
						assert(0);
					}
#endif
					result += dout[N * outputSize + yid] * input[N * inputSize + xid];
				}

				dA[id] = result;
				//printf("dA[%d]=%lf\n", id,result);
			}

			__global__ void biasBackward(f32* dBias, f32* dout, u32 outputSize, u32 batchSize)
			{
				u32 id = blockIdx.x * blockDim.x + threadIdx.x;
				if (id >= outputSize)
				{
					return;
				}
				f32 result = 0.0f;
				for (u32 N = 0; N < batchSize; N++)
				{
#if INDEX_DEBUG
					if ((N * outputSize + id) >= batchSize * outputSize)
					{
						assert(0);
					}
#endif
					result += dout[N * outputSize + id];
				}
#if INDEX_DEBUG
				if (id >= outputSize)
				{
					assert(0);
				}
#endif
				dBias[id] = result;
				//printf("%lf\n", result);
			}

			__global__ void doutBackward(f32* dOut, f32* A, f32* dIn, u32 outputSize, u32 inputSize, u32 batchSize)
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
#if INDEX_DEBUG
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
				//printf("dOut[%d * %d + %d] = %lf\n",yid, inputSize, xid, dOut[yid * inputSize + xid]);
			}
		}
		void Affine::mallocOnGPU()
		{
			mParametersPtrOnGPU.resize(2);
			mDParametersPtrOnGPU.resize(2);

			//Affineパラメータ
			DataArray& affineParam = mParametersPtrOnGPU[0];
			DataArray& affineDParam = mDParametersPtrOnGPU[0];

			affineParam.size = affineDParam.size = mOutputSize * mInputSize;

			CHECK(cudaMalloc((void**)(&(affineParam.address)), affineParam.size * sizeof(f32)));
			CHECK(cudaMalloc((void**)(&(affineDParam.address)), affineDParam.size * sizeof(f32)));

			{
				std::random_device seed_gen;
				std::default_random_engine engine(seed_gen());
				std::normal_distribution<> dist(0.0, std::sqrt(2.0 / mInputSize));

				std::vector<f32> tmp(affineParam.size);
				for (u32 idx = 0; idx < affineParam.size; idx++)
				{
					tmp[idx] = mAffineParamWeight * static_cast<f32>(dist(engine));
				}
				CHECK(cudaMemcpy(affineParam.address, tmp.data(), affineParam.size * sizeof(f32), cudaMemcpyHostToDevice));

				for (u32 idx = 0; idx < affineDParam.size; idx++)
				{
					tmp[idx] = 0.0f;
				}
				CHECK(cudaMemcpy(affineDParam.address, tmp.data(), affineDParam.size * sizeof(f32), cudaMemcpyHostToDevice));
			}


			//Biasパラメータ
			DataArray& biasParam = mParametersPtrOnGPU[1];
			DataArray& biasDParam = mDParametersPtrOnGPU[1];

			biasParam.size = biasDParam.size = mOutputSize;

			cudaMalloc((void**)(&(biasParam.address)), biasParam.size * sizeof(f32));
			cudaMalloc((void**)(&(biasDParam.address)), biasDParam.size * sizeof(f32));
			{
				f32* tmp = new f32[biasParam.size];
				for (u32 idx = 0; idx < biasParam.size; idx++)
				{
					tmp[idx] = 0.0f;
				}
				CHECK(cudaMemcpy(biasParam.address, tmp, biasParam.size * sizeof(f32), cudaMemcpyHostToDevice));
				CHECK(cudaMemcpy(biasDParam.address, tmp, biasDParam.size * sizeof(f32), cudaMemcpyHostToDevice));
				delete[] tmp;
			}

			//計算結果を格納するためのメモリ確保
			mForwardResultOnGPU.size = mBatchSize * mOutputSize;
			mBackwardResultOnGPU.size = mBatchSize * mInputSize;
			CHECK(cudaMalloc((void**)(&(mForwardResultOnGPU.address)),
				mForwardResultOnGPU.size * sizeof(f32)));
			CHECK(cudaMalloc((void**)(&(mBackwardResultOnGPU.address)),
				mBackwardResultOnGPU.size * sizeof(f32)));
			{
				f32* tmp = new f32[mForwardResultOnGPU.size];
				for (u32 idx = 0; idx < mForwardResultOnGPU.size; idx++)
				{
					tmp[idx] = 0.0f;
				}
				CHECK(cudaMemcpy(mForwardResultOnGPU.address, tmp,
					mForwardResultOnGPU.size * sizeof(f32), cudaMemcpyHostToDevice));
				delete[] tmp;


				tmp = new f32[mBackwardResultOnGPU.size];
				for (u32 idx = 0; idx < mBackwardResultOnGPU.size; idx++)
				{
					tmp[idx] = 0.0f;
				}
				CHECK(cudaMemcpy(mBackwardResultOnGPU.address, tmp,
					mBackwardResultOnGPU.size * sizeof(f32), cudaMemcpyHostToDevice));
				delete[] tmp;
			}
		}

		void Affine::forwardOnGPU()
		{
			//将来的にgroupSharedを利用したほうが早いと確定したらこの部分の後者で置き換える予定。
			//			{
			//				std::chrono::system_clock::time_point time = std::chrono::system_clock::now();
			//#if 0
			//				dim3 block(16, 16);
			//				dim3 grid(
			//					(mOutputSize + block.x - 1) / block.x,
			//					(mBatchSize + block.y - 1) / block.y);
			//
			//
			//				AffineForward << <grid, block >> > (
			//					mForwardResultOnGPU.address,
			//					mParametersPtrOnGPU[0].address,
			//					mInputDataOnGPU->address,
			//					mParametersPtrOnGPU[1].address,
			//					mOutputSize,
			//					mInputSize,
			//					mBatchSize);
			//
			//#if _DEBUG
			//				CHECK(cudaDeviceSynchronize());
			//#endif
			//
			//#else
			//				u32 sharedMemorySize = 48000 / sizeof(f32);
			//
			//
			//				const u32 BlockSize = std::min(static_cast<u32>(1 << 5), sharedMemorySize / (2 * mInputSize));
			//				if (BlockSize < 1)
			//				{
			//					std::cout << "BlockSize is less than 1\n";
			//					assert(0);
			//				}
			//				dim3 block(BlockSize, BlockSize);
			//				dim3 grid((mOutputSize + BlockSize - 1) / BlockSize, (mBatchSize + BlockSize - 1) / BlockSize);
			//
			//				AffineForwardWithSM << <grid, block, 2 * mInputSize * BlockSize * sizeof(f32) >> > (
			//					mForwardResultOnGPU.address,
			//					mParametersPtrOnGPU[0].address,
			//					mInputDataOnGPU->address,
			//					mParametersPtrOnGPU[1].address,
			//					mOutputSize,
			//					mInputSize,
			//					mBatchSize);
			//
			//#if _DEBUG
			//				CHECK(cudaDeviceSynchronize());
			//#endif
			//
			//#endif
			//				auto time2 = static_cast<f32>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - time).count() / 1000.0f);
			//				//std::cout << time2 << std::endl;
			//				return;
			//		}


			std::chrono::system_clock::time_point time;


			if (mWhich == 0)
			{
				if (mNowComparing)
				{
					time = std::chrono::system_clock::now();
				}
				dim3 block(32, 32);
				dim3 grid(
					(mOutputSize + block.x - 1) / block.x,
					(mBatchSize + block.y - 1) / block.y);

#if TIME_DEBUG
				{
					std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
					AffineForward << <grid, block >> > (
						mForwardResultOnGPU.address,
						mParametersPtrOnGPU[0].address,
						mInputDataOnGPU->address,
						mParametersPtrOnGPU[1].address,
						mOutputSize,
						mInputSize,
						mBatchSize);
#if GPU_SYNC_DEBUG
					CHECK(cudaDeviceSynchronize());
#endif
#if TIME_DEBUG
					f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
					std::string name = "";
					(((name += __FUNCTION__) += " : ") += std::to_string(mInstanceID)) += " : AffineForward";
					timers[name] = elapsedTime;
				}
#endif
				if (mNowComparing)
				{
#if !GPU_SYNC_DEBUG
					CHECK(cudaDeviceSynchronize());
#endif
					auto elapsedTime = static_cast<f32>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - time).count() / 1000.0f);

					mFunc0AveTime = (mFunc0AveTime * mFunc0CallCnt + elapsedTime) / (mFunc0CallCnt + 1);
					mFunc0CallCnt++;

					if (mFunc0CallCnt >= CaptureTimes)
					{
						mWhich = 1;
					}
				}
			}
			else if (mWhich = 1)
			{
				if (mNowComparing)
				{
					time = std::chrono::system_clock::now();
				}

				u32 sharedMemorySize = 48000 / sizeof(f32);


				const u32 BlockSize = std::min(static_cast<u32>(1 << 5), sharedMemorySize / (2 * mInputSize));
				if (BlockSize < 1)
				{
					std::cout << "BlockSize is less than 1\n";
					assert(0);
				}
				dim3 block(BlockSize, BlockSize);
				dim3 grid((mOutputSize + BlockSize - 1) / BlockSize, (mBatchSize + BlockSize - 1) / BlockSize);
#if TIME_DEBUG
				{
					std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
					AffineForwardWithSM << <grid, block, 2 * mInputSize * BlockSize * sizeof(f32) >> > (
						mForwardResultOnGPU.address,
						mParametersPtrOnGPU[0].address,
						mInputDataOnGPU->address,
						mParametersPtrOnGPU[1].address,
						mOutputSize,
						mInputSize,
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
				if (mNowComparing)
				{
#if !GPU_SYNC_DEBUG
					CHECK(cudaDeviceSynchronize());
#endif
					auto elapsedTime = static_cast<f32>(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - time).count() / 1000.0f);

					mFunc1AveTime = (mFunc1AveTime * mFunc1CallCnt + elapsedTime) / (mFunc1CallCnt + 1);
					mFunc1CallCnt++;

					if (mFunc1CallCnt >= CaptureTimes)
					{
						if (mFunc1AveTime > mFunc0AveTime)
						{
							mWhich = 0;
						}
						else
						{
							mWhich = 1;
						}
						mNowComparing = false;
					}
				}

			}
			else
			{
				assert(0);
			}
		}

		void Affine::backwardOnGPU()
		{
			//doutの逆伝搬
			{
				dim3 block(16, 16);
				dim3 grid(
					(mInputSize + block.x - 1) / block.x,
					(mBatchSize + block.y - 1) / block.y);
#if TIME_DEBUG
				{
					std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
					doutBackward << <grid, block >> > (
						mBackwardResultOnGPU.address,
						mParametersPtrOnGPU[0].address,
						mDInputDataOnGPU->address,
						mOutputSize,
						mInputSize,
						mBatchSize);
#if GPU_SYNC_DEBUG
					CHECK(cudaDeviceSynchronize());
#endif
#if TIME_DEBUG
					f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
					std::string name = "";
					(((name += __FUNCTION__) += " : ") += std::to_string(mInstanceID)) += " : doutBackward";
					timers[name] = elapsedTime;
				}
#endif
			}

			//Aの逆伝搬
			{
				dim3 block(16, 16);
				dim3 grid(
					(mInputSize + block.x - 1) / block.x,
					(mOutputSize + block.y - 1) / block.y);
#if TIME_DEBUG
				{
					std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
					AffineBackward << <grid, block >> > (
						mDParametersPtrOnGPU[0].address,
						mDInputDataOnGPU->address,
						mInputDataOnGPU->address,
						mOutputSize,
						mInputSize,
						mBatchSize);

#if GPU_SYNC_DEBUG
					CHECK(cudaDeviceSynchronize());
#endif
#if TIME_DEBUG
					f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
					std::string name = "";
					(((name += __FUNCTION__) += " : ") += std::to_string(mInstanceID)) += " : AffineBackward";
					timers[name] = elapsedTime;
				}
#endif
			}

			//Biasの逆伝搬
			{
				dim3 block(16);
				dim3 grid((mOutputSize + block.x - 1) / block.x);
#if TIME_DEBUG
				{
					std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
					biasBackward << <grid, block >> > (
						mDParametersPtrOnGPU[1].address,
						mDInputDataOnGPU->address,
						mOutputSize,
						mBatchSize);

#if GPU_SYNC_DEBUG
					CHECK(cudaDeviceSynchronize());
#endif
#if TIME_DEBUG
					f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
					std::string name = "";
					(((name += __FUNCTION__) += " : ") += std::to_string(mInstanceID)) += " : biasBackward";
					timers[name] = elapsedTime;
				}
#endif
			}
		}

		void Affine::terminateOnGPU()
		{

		}

	}
}