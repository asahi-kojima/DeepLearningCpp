#include <random>
#include <cuda_runtime.h>
#include <cassert>

#include "BatchNorm2d.h"

namespace Aoba {
	namespace layer
	{
		namespace
		{
			__global__ void BatchNorm2d_computeBlockMeans(
				f32* input,
				f32* blockMean,
				f32* blockSqMean,
				u32 batchSize,
				u32 channel,
				u32 height,
				u32 width)
			{
				u32 Ic = blockIdx.x * blockDim.x + threadIdx.x;
				u32 IhIw = blockIdx.y * blockDim.y + threadIdx.y;

				if (Ic >= channel || IhIw >= height * width)
				{
					return;
				}

				u32 mIhIw = height * width;
				u32 mIcIhIw = channel * mIhIw;

				f32 mean = 0.0f;
				f32 sqMean = 0.0f;

				//------------------------------------------------------------------
				//ïΩãœÇåvéZ
				//------------------------------------------------------------------
				for (u32 N = 0; N < batchSize; N++)
				{
					f32 value = input[N * mIcIhIw + Ic * mIhIw + IhIw];
					mean += value;
					sqMean += value * value;
				}

				blockMean[Ic * mIhIw + IhIw] = mean;
				blockSqMean[Ic * mIhIw + IhIw] = sqMean;
			}

			__global__ void BatchNorm2d_computeMeanSigma(
				f32* blockMean,
				f32* blockSqMean,
				f32* Mean,
				f32* Sigma,
				u32 batchSize,
				u32 channel,
				u32 height,
				u32 width)
			{
				u32 Ic = blockIdx.x * blockDim.x + threadIdx.x;

				if (Ic >= channel)
				{
					return;
				}

				f32 ep = 1e-7;
				u32 mIhIw = height * width;

				f32 mean = 0.0f;
				f32 sqMean = 0.0f;

				//------------------------------------------------------------------
				//ïΩãœÇåvéZ
				//------------------------------------------------------------------
				for (u32 IhIw = 0; IhIw < mIhIw; IhIw++)
				{
					mean += blockMean[Ic * mIhIw + IhIw];
					sqMean += blockSqMean[Ic * mIhIw + IhIw];
				}

				mean /= (batchSize * mIhIw);
				sqMean /= (batchSize * mIhIw);

				//------------------------------------------------------------------
				//ïŒç∑ÇåvéZ
				//------------------------------------------------------------------
				Mean[Ic] = mean;
				Sigma[Ic] = std::sqrt(sqMean - mean * mean) + ep;
			}

			__global__ void BatchNorm2d_forwardOnGPU(
				f32* input,
				f32* intermediateResult,
				f32* forwardResult,
				f32* Gamma,
				f32* Beta,
				f32* Mean,
				f32* Sigma,
				u32 batchSize,
				u32 channel,
				u32 height,
				u32 width)
			{
				u32 Ic = blockIdx.x * blockDim.x + threadIdx.x;
				u32 IhIw = blockIdx.y * blockDim.y + threadIdx.y;

				if (Ic >= channel || IhIw >= height * width)
				{
					return;
				}


				u32 mIhIw = height * width;
				u32 mIcIhIw = channel * mIhIw;

				f32 mean = Mean[Ic];
				f32 sigma = Sigma[Ic];


				//------------------------------------------------------------------
				//ïWèÄâª
				//------------------------------------------------------------------
				f32 gamma = Gamma[Ic];
				f32 beta = Beta[Ic];
				for (u32 N = 0; N < batchSize; N++)
				{
					u32 index = N * mIcIhIw + Ic * mIhIw + IhIw;
					f32 normalizeResult = (input[index] - mean) / sigma;
					intermediateResult[index] = normalizeResult;
					forwardResult[index] = gamma * normalizeResult + beta;
				}
			}


			__global__ void BatchNorm2d_backwardOnGPU_computeBlock(
				f32* dout,
				f32* intermediateResult,
				f32* dBlockGamma,
				f32* dBlockBeta,
				f32* dBlockMean,
				f32* dBlockIMean,
				u32 batchSize,
				u32 channel,
				u32 height,
				u32 width)
			{
				u32 Ic = blockIdx.x * blockDim.x + threadIdx.x;
				u32 IhIw = blockIdx.y * blockDim.y + threadIdx.y;

				if (Ic >= channel || IhIw >= height * width)
				{
					return;
				}

				f32 ep = 1e-7;
				u32 mIhIw = height * width;
				u32 mIcIhIw = channel * mIhIw;

				f32 dGamma = 0.0f;
				f32 dBeta = 0.0f;
				f32 dMean = 0.0f;
				f32 dIMean = 0.0f;

				for (u32 N = 0; N < batchSize; N++)
				{
					u32 index = N * mIcIhIw + Ic * mIhIw + IhIw;
					f32 dO = dout[index];
					f32 iR = intermediateResult[index];

					dGamma += dO * iR;
					dBeta += dO;
					dMean += dO;
					dIMean += dO * iR;
				}

				dBlockGamma[Ic * mIhIw + IhIw] = dGamma;
				dBlockBeta[Ic * mIhIw + IhIw] = dBeta;
				dBlockMean[Ic * mIhIw + IhIw] = dMean;
				dBlockIMean[Ic * mIhIw + IhIw] = dIMean;
			}

			__global__ void BatchNorm2d_backwardOnGPU_computeDValue(
				f32* dGamma,
				f32* dBlockGamma,
				f32* dBeta,
				f32* dBlockBeta,
				f32* dMean,
				f32* dBlockMean,
				f32* dIMean,
				f32* dBlockIMean,
				u32 batchSize,
				u32 channel,
				u32 height,
				u32 width)
			{
				u32 Ic = blockIdx.x * blockDim.x + threadIdx.x;

				if (Ic >= channel)
				{
					return;
				}

				f32 ep = 1e-7;
				u32 mIhIw = height * width;
				u32 mIcIhIw = channel * mIhIw;

				f32 gamma = 0.0f;
				f32 beta = 0.0f;
				f32 mean = 0.0f;
				f32 iMean = 0.0f;

				for (u32 IhIw = 0; IhIw < mIhIw; IhIw++)
				{
					u32 index = Ic * mIhIw + IhIw;

					gamma += dBlockGamma[index];
					beta += dBlockBeta[index];
					mean += dBlockMean[index];
					iMean += dBlockIMean[index];
				}

				dGamma[Ic] = gamma;
				dBeta[Ic] = beta;
				dMean[Ic] = mean / (batchSize * mIhIw);
				dIMean[Ic] = iMean / (batchSize * mIhIw);
			}

			__global__ void BatchNorm2d_backwardOnGPU(
				f32* dout,
				f32* intermediateResult,
				f32* backwardResult,
				f32* Gamma,
				f32* DGamma,
				f32* DBeta,
				f32* Sigma,
				u32 batchSize,
				u32 channel,
				u32 height,
				u32 width)
			{
				u32 c = blockIdx.x * blockDim.x + threadIdx.x;

				if (c >= channel)
				{
					return;
				}



				f32 ep = 1e-7;
				u32 hXw = height * width;
				u32 cXhXw = channel * hXw;

				f32 dGamma = 0.0f;
				f32 dBeta = 0.0f;

				for (u32 N = 0; N < batchSize; N++)
				{
					for (u32 hw = 0; hw < hXw; hw++)
					{
						u32 index = N * cXhXw + c * hXw + hw;
						f32 dO = dout[index];
						dGamma += dO * intermediateResult[index];
						dBeta += dO;
					}
				}
				DGamma[c] = dGamma;
				DBeta[c] = dBeta;



				f32 dMean = 0.0f;
				f32 diMean = 0.0f;
				for (u32 N = 0; N < batchSize; N++)
				{
					for (u32 hw = 0; hw < hXw; hw++)
					{
						u32 index = N * cXhXw + c * hXw + hw;
						dMean += dout[index];
						diMean += dout[index] * intermediateResult[index];
					}
				}
				dMean /= (batchSize * hXw);
				diMean /= (batchSize * hXw);

				for (u32 N = 0; N < batchSize; N++)
				{
					for (u32 hw = 0; hw < hXw; hw++)
					{
						u32 index = N * cXhXw + c * hXw + hw;
						backwardResult[index] = (Gamma[c] / (Sigma[c] + 1e-7)) * (dout[index] - dMean - intermediateResult[index] * diMean);
					}
				}
			}

			__global__ void BatchNorm2d_backwardOnGPU2(
				f32* dout,
				f32* intermediateResult,
				f32* backwardResult,
				f32* Gamma,
				f32* Sigma,
				f32* DMean,
				f32* DIMean,
				u32 batchSize,
				u32 channel,
				u32 height,
				u32 width)
			{
				u32 N = blockIdx.x * blockDim.x + threadIdx.x;
				u32 IcIhIw = blockIdx.y * blockDim.y + threadIdx.y;

				const u32 mIcIhIw = channel * height * width;
				if (N >= batchSize || IcIhIw >= mIcIhIw)
				{
					return;
				}

				const u32 mIhIw = height * width;
				const u32 Ic = IcIhIw / mIhIw;

				const u32 index = N * mIcIhIw + IcIhIw;

				backwardResult[index]
					= 
					(Gamma[Ic] / (Sigma[Ic] + 1e-7)) * (dout[index] - DMean[Ic] - intermediateResult[index] * DIMean[Ic]);
			}
		}


		void BatchNorm2d::mallocOnGPU()
		{
			mParametersPtrOnGPU.resize(2);
			mDParametersPtrOnGPU.resize(2);


			//------------------------------------------------------------------
			//Gamma
			//------------------------------------------------------------------
			DataArray& gammaParam = mParametersPtrOnGPU[0];
			DataArray& gammaDParam = mDParametersPtrOnGPU[0];

			gammaParam.size = mDataShape.channel;
			gammaDParam.size = gammaParam.size;

			CHECK(cudaMalloc((void**)(&(gammaParam.address)), gammaParam.size * sizeof(f32)));
			CHECK(cudaMalloc((void**)(&(gammaDParam.address)), gammaDParam.size * sizeof(f32)));
			{
				f32* tmp = new f32[gammaParam.size];
				for (u32 i = 0; i < gammaParam.size; i++)
				{
					tmp[i] = 1.0f;
				}
				CHECK(cudaMemcpy(gammaParam.address, tmp, gammaParam.size * sizeof(f32), cudaMemcpyHostToDevice));
				CHECK(cudaMemcpy(gammaDParam.address, tmp, gammaDParam.size * sizeof(f32), cudaMemcpyHostToDevice));
				delete[] tmp;
			}


			//------------------------------------------------------------------
			//Beta
			//------------------------------------------------------------------
			DataArray& betaParam = mParametersPtrOnGPU[1];
			DataArray& betaDParam = mDParametersPtrOnGPU[1];

			betaParam.size = mDataShape.channel;
			betaDParam.size = betaParam.size;

			CHECK(cudaMalloc((void**)(&(betaParam.address)), betaParam.size * sizeof(f32)));
			CHECK(cudaMalloc((void**)(&(betaDParam.address)), betaDParam.size * sizeof(f32)));

			{
				f32* tmp = new f32[betaParam.size];
				for (u32 i = 0; i < betaParam.size; i++)
				{
					tmp[i] = 0.0f;
				}
				CHECK(cudaMemcpy(betaParam.address, tmp, betaParam.size * sizeof(f32), cudaMemcpyHostToDevice));
				CHECK(cudaMemcpy(betaDParam.address, tmp, betaDParam.size * sizeof(f32), cudaMemcpyHostToDevice));
				delete[] tmp;
			}

			//------------------------------------------------------------------
			//Sigma
			//------------------------------------------------------------------
			mSigmaOnGPU.size = mDataShape.channel;
			mSigmaOnGPU.byteSize = mSigmaOnGPU.size * sizeof(f32);

			CHECK(cudaMalloc((void**)(&(mSigmaOnGPU.address)), mSigmaOnGPU.size * sizeof(f32)));

			{
				f32* tmp = new f32[mSigmaOnGPU.size];
				for (u32 i = 0; i < mSigmaOnGPU.size; i++)
				{
					tmp[i] = 0.0f;
				}
				CHECK(cudaMemcpy(mSigmaOnGPU.address, tmp, mSigmaOnGPU.size * sizeof(f32), cudaMemcpyHostToDevice));
				delete[] tmp;
			}

			mMeanOnGPU.size = mDataShape.channel;
			MALLOC_AND_INITIALIZE_0_ON_GPU(mMeanOnGPU);

			mBlockSqMeanOnGPU.size = mDataShape.getDataSize();
			MALLOC_AND_INITIALIZE_0_ON_GPU(mBlockSqMeanOnGPU);
			mBlockMeanOnGPU.size = mDataShape.getDataSize();
			MALLOC_AND_INITIALIZE_0_ON_GPU(mBlockMeanOnGPU);

			mDMeanOnGPU.size = mDataShape.channel;
			MALLOC_AND_INITIALIZE_0_ON_GPU(mDMeanOnGPU);
			mDIMeanOnGPU.size = mDataShape.channel;
			MALLOC_AND_INITIALIZE_0_ON_GPU(mDIMeanOnGPU);

			mBlockDMeanOnGPU.size = mDataShape.getDataSize();
			MALLOC_AND_INITIALIZE_0_ON_GPU(mBlockDMeanOnGPU);
			mBlockDIMeanOnGPU.size = mDataShape.getDataSize();
			MALLOC_AND_INITIALIZE_0_ON_GPU(mBlockDIMeanOnGPU);
			mBlockDGammaOnGPU.size = mDataShape.getDataSize();
			MALLOC_AND_INITIALIZE_0_ON_GPU(mBlockDGammaOnGPU);
			mBlockDBetaOnGPU.size = mDataShape.getDataSize();
			MALLOC_AND_INITIALIZE_0_ON_GPU(mBlockDBetaOnGPU);
			//------------------------------------------------------------------
			//ì`î¿óp
			//------------------------------------------------------------------
			mForwardResultOnGPU.size = mBatchSize * mDataShape.getDataSize();
			mBackwardResultOnGPU.size = mForwardResultOnGPU.size;
			mIntermediateResultOnGPU.size = mForwardResultOnGPU.size;

			mForwardResultOnGPU.address = new f32[mForwardResultOnGPU.size];
			mBackwardResultOnGPU.address = new f32[mBackwardResultOnGPU.size];
			mIntermediateResultOnGPU.address = new f32[mIntermediateResultOnGPU.size];
			CHECK(cudaMalloc((void**)(&(mForwardResultOnGPU.address)), mForwardResultOnGPU.size * sizeof(f32)));
			CHECK(cudaMalloc((void**)(&(mBackwardResultOnGPU.address)), mBackwardResultOnGPU.size * sizeof(f32)));
			CHECK(cudaMalloc((void**)(&(mIntermediateResultOnGPU.address)), mIntermediateResultOnGPU.size * sizeof(f32)));

			{
				f32* tmp = new f32[mForwardResultOnGPU.size];
				for (u32 i = 0; i < mForwardResultOnGPU.size; i++)
				{
					tmp[i] = 0.0f;
				}
				CHECK(cudaMemcpy(mForwardResultOnGPU.address, tmp, mForwardResultOnGPU.size * sizeof(f32), cudaMemcpyHostToDevice));
				CHECK(cudaMemcpy(mBackwardResultOnGPU.address, tmp, mBackwardResultOnGPU.size * sizeof(f32), cudaMemcpyHostToDevice));
				CHECK(cudaMemcpy(mIntermediateResultOnGPU.address, tmp, mIntermediateResultOnGPU.size * sizeof(f32), cudaMemcpyHostToDevice));
				delete[] tmp;
			}
		}

		void BatchNorm2d::forwardOnGPU()
		{
			{
				dim3 block(16, 16);
				dim3 grid(
					(mDataShape.channel + block.x - 1) / block.x,
					((mDataShape.height * mDataShape.width) + block.y - 1) / block.y);
#if TIME_DEBUG
				{
					std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
					BatchNorm2d_computeBlockMeans << <grid, block >> >
						(
							mInputDataOnGPU->address,
							mBlockMeanOnGPU.address,
							mBlockSqMeanOnGPU.address,
							mBatchSize,
							mDataShape.channel,
							mDataShape.height,
							mDataShape.width
							);

#if GPU_SYNC_DEBUG
					CHECK(cudaDeviceSynchronize());
#endif
#if TIME_DEBUG
					f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
					std::string name = "";
					(((name += __FUNCTION__) += " : ") += std::to_string(mInstanceID)) += " : computeBlockMean";
					timers[name] = elapsedTime;
				}
#endif
			}

			{
				dim3 block(16);
				dim3 grid(
					(mDataShape.channel + block.x - 1) / block.x);
#if TIME_DEBUG
				{
					std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
					BatchNorm2d_computeMeanSigma << <grid, block >> >
						(
							mBlockMeanOnGPU.address,
							mBlockSqMeanOnGPU.address,
							mMeanOnGPU.address,
							mSigmaOnGPU.address,
							mBatchSize,
							mDataShape.channel,
							mDataShape.height,
							mDataShape.width
							);

#if GPU_SYNC_DEBUG
					CHECK(cudaDeviceSynchronize());
#endif
#if TIME_DEBUG
					f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
					std::string name = "";
					(((name += __FUNCTION__) += " : ") += std::to_string(mInstanceID)) += " : ComputeMeanSigma";
					timers[name] = elapsedTime;
				}
#endif
			}

			{
				dim3 block(16, 16);
				dim3 grid(
					(mDataShape.channel + block.x - 1) / block.x,
					((mDataShape.height * mDataShape.width) + block.y - 1) / block.y);
#if TIME_DEBUG
				{
					std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
					BatchNorm2d_forwardOnGPU << <grid, block >> > (
						mInputDataOnGPU->address,
						mIntermediateResultOnGPU.address,
						mForwardResultOnGPU.address,
						mParametersPtrOnGPU[0].address,
						mParametersPtrOnGPU[1].address,
						mMeanOnGPU.address,
						mSigmaOnGPU.address,
						mBatchSize,
						mDataShape.channel,
						mDataShape.height,
						mDataShape.width);
#if GPU_SYNC_DEBUG
					CHECK(cudaDeviceSynchronize());
#endif
#if TIME_DEBUG
					f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
					std::string name = "";
					(((name += __FUNCTION__) += " : ") += std::to_string(mInstanceID)) += " : forward";
					timers[name] = elapsedTime;
				}
#endif
			}
		}

		void BatchNorm2d::backwardOnGPU()
		{
			{
				dim3 block(16, 16);
				dim3 grid(
					(mDataShape.channel + block.x - 1) / block.x,
					((mDataShape.height * mDataShape.width) + block.y - 1) / block.y);
#if TIME_DEBUG
				{
					std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
					BatchNorm2d_backwardOnGPU_computeBlock << <grid, block >> > (
						mDInputDataOnGPU->address,
						mIntermediateResultOnGPU.address,
						mBlockDGammaOnGPU.address,
						mBlockDBetaOnGPU.address,
						mBlockDMeanOnGPU.address,
						mBlockDIMeanOnGPU.address,
						mBatchSize,
						mDataShape.channel,
						mDataShape.height,
						mDataShape.width);

#if GPU_SYNC_DEBUG
					CHECK(cudaDeviceSynchronize());
#endif
#if TIME_DEBUG
					f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
					std::string name = "";
					(((name += __FUNCTION__) += " : ") += std::to_string(mInstanceID)) += " : computeBlock";
					timers[name] = elapsedTime;
				}
#endif
			}

			{
				dim3 block(16);
				dim3 grid((mDataShape.channel + block.x - 1) / block.x);
#if TIME_DEBUG
				{
					std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
					BatchNorm2d_backwardOnGPU_computeDValue << <grid, block >> > (
						mDParametersPtrOnGPU[0].address,
						mBlockDGammaOnGPU.address,
						mDParametersPtrOnGPU[1].address,
						mBlockDBetaOnGPU.address,
						mDMeanOnGPU.address,
						mBlockDMeanOnGPU.address,
						mDIMeanOnGPU.address,
						mBlockDIMeanOnGPU.address,
						mBatchSize,
						mDataShape.channel,
						mDataShape.height,
						mDataShape.width);

#if GPU_SYNC_DEBUG
					CHECK(cudaDeviceSynchronize());
#endif
#if TIME_DEBUG
					f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
					std::string name = "";
					(((name += __FUNCTION__) += " : ") += std::to_string(mInstanceID)) += " : computeDValue";
					timers[name] = elapsedTime;
				}
#endif
			}

			{
				dim3 block(16, 16);
				dim3 grid(
					(mBatchSize + block.x - 1) / block.x,
					(mDataShape.getDataSize() + block.x - 1) / block.y);
#if TIME_DEBUG
				{
					std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
#endif
					BatchNorm2d_backwardOnGPU2 << <grid, block >> > (
						mDInputDataOnGPU->address,
						mIntermediateResultOnGPU.address,
						mBackwardResultOnGPU.address,
						mParametersPtrOnGPU[0].address,
						mSigmaOnGPU.address,
						mDMeanOnGPU.address,
						mDIMeanOnGPU.address,
						mBatchSize,
						mDataShape.channel,
						mDataShape.height,
						mDataShape.width);

#if GPU_SYNC_DEBUG
					CHECK(cudaDeviceSynchronize());
#endif
#if TIME_DEBUG
					f32 elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count() / 1000.0f;
					std::string name = "";
					(((name += __FUNCTION__) += " : ") += std::to_string(mInstanceID)) += " : backward";
					timers[name] = elapsedTime;
				}
#endif
			}
		}

		void BatchNorm2d::terminateOnGPU()
		{
			for (u32 id = 0; id < mParametersPtrOnGPU.size(); id++)
			{
				delete[] mParametersPtrOnGPU[id].address;
				delete[] mDParametersPtrOnGPU[id].address;
			}

			delete[] mSigmaOnGPU.address;

			delete[] mForwardResultOnGPU.address;
			delete[] mBackwardResultOnGPU.address;
			delete[] mIntermediateResultOnGPU.address;
		}
	}
}