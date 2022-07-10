#include "Convolution.hpp"
#include <time.h>

//******************************
//       初期化関数
//******************************
void Convolution::initialize()
{
}

//******************************
//       順伝搬関数　実装
//******************************
void Convolution::forward(cFlowType * & pInput)
{
    cFlowType & input = *pInput;
    for (S32 N = 0; N < mBatchNum; N++)
    {
        mInputData[N] = input[N];
        for (S32 i = 0; i < mOhOw * mIcFhFw; i++)
        {
            S32 V = i / mIcFhFw;
            S32 H = i % mIcFhFw;

            S32 Fh = V / mOutputWidth;
            S32 Fw = V % mOutputWidth;

            S32 c = H / mFhFw;
            S32 h = (H - c * mFhFw) / mFilterWidth;
            S32 w = H % mFilterWidth;
            mReshapedInputData[N][i] = 
            ((w - mPad < 0 || w >= mInputWidth + mPad || h - mPad < 0 || h >= mInputHeight + mPad) ? 
            0 : mInputData[N][c * mIhIw + (Fh * mStride + (h - mPad)) * mInputWidth + (Fw * mStride + (w - mPad))]);
        }

        for (S32 i = 0; i < mOcOhOw; i++)
        {
            F32 tmp = 0;
            S32 Fc = i / mOhOw;
            S32 i1 = i % mOhOw;

            for (S32 k = 0; k < mIcFhFw; k++)
            {
                tmp += mMatrix[Fc * mIcFhFw + k] * mReshapedInputData[N][i1 * mIcFhFw + k];
            }
            mForwardResult[N][i] = tmp + mbias[i / mOhOw];
        }
    }

    input = &mForwardResult;
}

//******************************
//       逆伝搬関数　実装
//******************************
void Convolution::backward(cFlowType * & pDout)
{
    cFlowType & dout = *pDout;
    for (S32 i = 0; i < mOutputChannel * mIcFhFw; i++)
    {
        F32 tmp = 0;
        S32 x = i / mIcFhFw;
        S32 y = i % mIcFhFw;
        for (S32 N = 0; N < mBatchNum; N++)
        {
            for (S32 k = 0; k < mOhOw; k++)
            {
                tmp += dout[N][x * mOhOw + k] * mReshapedInputData[N][k * mIcFhFw + y];
            } 
        }
        mDMatrix[i] = tmp;
        //std::cout << "mat = " << tmp << std::endl;
    }
    //std::cout << "test-back1" << std::endl;
    for (S32 i = 0; i < mOutputChannel; i++)
    {
        F32 tmp = 0;
        for (S32 N = 0; N < mBatchNum; N++)
        {
            for (S32 j = 0; j < mOhOw; j++)
            {
                tmp += dout[N][i * mOhOw + j];
            }
        }
        mDBias[i] = tmp;
        //std::cout << "bias = " << tmp << std::endl;
    }




#ifdef _OPENMP
    #pragma omp pallel for
#endif
    for (S32 N = 0; N < mBatchNum; N++)
    {
        for (S32 i = 0; i < mIcIhIw; i++)
        {
            mBackwardResult[N][i] = 0.0;
        }

        //F32 dL7dI[mOhOw * mIcFhFw];

        for (S32 i = 0; i < mOhOw * mIcFhFw; i++)
        {
            F32 tmp = 0;
            for (S32 j = 0; j < mFilterNum; j++)
            {
                S32 i1 = i / mIcFhFw;
                S32 i2 = i % mIcFhFw;
                tmp += dout[N][j * mOhOw + i1] * mMatrix[j * mIcFhFw + i2];
            }
            //dL7dI[i] = tmp;

            S32 OhOw = i / mIcFhFw;
            S32 fCol = OhOw / mOutputWidth;
            S32 fRow = OhOw % mOutputWidth;

            S32 iRes = i - mIcFhFw * OhOw;
            S32 c = iRes / mFhFw;
            S32 h = (iRes - c * mFhFw) / mFilterWidth; 
            S32 w = i % mFilterWidth;
            if (h - mPad < 0 || h - mPad >= mInputHeight || w - mPad < 0 || w - mPad >= mInputWidth)
            {
                continue;
            }
            mBackwardResult[N][c * mIhIw + (fCol * mStride + h - mPad) * mInputWidth + (fRow * mStride + w - mPad)] 
            +=
            tmp;
        }
    }

    pDout = &mBackwardResult;
}

