#include "Deconvolution.hpp"
#include <time.h>

//******************************
//       初期化関数
//******************************
void Deconvolution::initialize()
{
}

void Deconvolution::forward(cFlowType * & pInput)
{
    cFlowType & input = *pInput;
    static bool isAlreadyCalc = false;
    for (S32 N = 0; N < mBatchNum; N++)
    {
        for (int _IcIhIw = 0; _IcIhIw < mIcIhIw; _IcIhIw++)
        {
            for (int index = 0; index < mProjectionIndex[N][_IcIhIw]; index++)
            {
                mReshapedInputData[N][mProjectionList[N][_IcIhIw][index]] = input[N][_IcIhIw];
            }
        }
        

        for (S32 _OcOhOw = 0; _OcOhOw < mOcOhOw; _OcOhOw++)
        {
            S32 _Oc = _OcOhOw / mOhOw;
            S32 _OhOw = _OcOhOw - _Oc * mOhOw;

            F32 tmp = 0;
            for (S32 k = 0; k < mFcFhFw; k++)
            {

                tmp += mMatrix[_Oc * mFcFhFw + k] * mReshapedInputData[N][_OhOw * mFcFhFw + k];
            }
            
            mForwardResult[N][_OcOhOw] = tmp + mBias[_Oc];
        }
    }

    isAlreadyCalc = true;

    pInput = &mForwardResult;
}


//******************************
//       逆伝搬関数　実装
//******************************
void Deconvolution::backward(cFlowType * & pdout)
{
    cFlowType & dout = *pdout;

    for (S32 _Oc = 0; _Oc < mOc; _Oc++)
    {
        F32 tmp = 0;
        for (S32 N = 0; N < mBatchNum; N++)
        {
            for (S32 _OhOw = 0; _OhOw < mOhOw; _OhOw++)
            {

                tmp += dout[N][_Oc * mOhOw + _OhOw];
            }
        }
    
        mDBias[_Oc] = tmp;
    }



    for (S32 index = 0; index < mFn * mFcFhFw; index++)
    {
        F32 tmp = 0;
        S32 x = index / mFcFhFw;
        S32 y = index % mFcFhFw;
        for (S32 N = 0; N < mBatchNum; N++)
        {
            for (S32 _OhOw = 0; _OhOw < mOhOw; _OhOw++)
            {

                tmp += dout[N][x * mOhOw + _OhOw] * mReshapedInputData[N][_OhOw * mFcFhFw + y];
            } 
        }

        mDMatrix[index] = tmp;
    }



    for (S32 N = 0; N < mBatchNum; N++)
    {
        for (S32 _IcIhIw = 0; _IcIhIw < mIcIhIw; _IcIhIw++)
        {
            F32 tmp = 0;
            for (S32 index = 0; index < mProjectionIndex[N][_IcIhIw]; index++)
            {

                S32 Index = mProjectionList[N][_IcIhIw][index];
                S32 IndexH = Index / mFcFhFw;
                S32 IndexW = Index % mFcFhFw;
                for (S32 _Fn = 0; _Fn < mFn; _Fn++)
                {

                    tmp += dout[N][_Fn * mOhOw + IndexH] * mMatrix[_Fn * mFcFhFw + IndexW];
                }
            }
            mBackwardResult[N][_IcIhIw] = tmp;
        }
    }

    pdout = &mBackwardResult;
}


void Deconvolution::calcIndexList()
{
    S32 Ph = mFh - 1 - mPad;
    S32 Pw = mFw - 1 - mPad;
    for (S32 N = 0; N < mBatchNum; N++)
    {
        for (S32 _IcIhIw = 0; _IcIhIw < mIcIhIw; _IcIhIw++)
        {
            S32 _Ic = _IcIhIw / mIhIw;
            S32 _Ih = (_IcIhIw - _Ic * mIhIw) / mIw;
            S32 _Iw = _IcIhIw % mIh;

            //拡張した行列における位置・インデックスを表す。
            S32 startPointFH = Ph + _Ih * mStride;
            S32 startPointFW = Pw + _Iw * mStride;


            for (S32 _oh = 0; _oh < mOh; _oh++)
            {
                if (startPointFH >= _oh && startPointFH < mFh + _oh)
                {
                    for (S32 _ow = 0; _ow < mOw; _ow++)
                    {
                        if (startPointFW >= _ow && startPointFW < mFw + _ow)
                        {
                            S32 index = (_oh * mOw + _ow) * mFcFhFw + (_Ic) * mFhFw + (startPointFW - _ow) * mFw + (startPointFH - _oh);
                            mProjectionList[N][_IcIhIw][mProjectionIndex[N][_IcIhIw]] = index;
                            mProjectionIndex[N][_IcIhIw]++; 
                        }
                    }
                }
            }
        }
    }
}

