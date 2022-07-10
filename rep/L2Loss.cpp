#include "L2Loss.hpp"



void L2Loss::initialize()
{

}

void L2Loss::forward(cFlowType * & input)
{
    mBatchLoss = 0;

    for (S32 N = 0; N < mBatchSize; N++)
    {
        mForwardResult[N][0] = 0;
        for (S32 I = 0; I < mDataSize; I++)
        {
            F32 diff = ((*input)[N][I] - (*mTarget)[N][I]);
            mIntermediumResult[N][I] = diff;
            F32 result = 0.5 * diff * diff;
            mForwardResult[N][0] += result;
            mBatchLoss += result;
        }
        mBatchLoss += mForwardResult[N][0];
    }

    mBatchLoss /= mBatchSize;

    input = &mForwardResult;
}


void L2Loss::backward(cFLowType * & dout)
{
    for (S32 N = 0; N < mBatchSize; N++)
    {
        for (S32 I = 0; I < mDataSize; I++)
        {
            mBackwardResult[N][I] =  mIntermediumResult[N][I];//forwardで代入すると計算量削減になる。
        }
    }

    dout = &mBackwardResult;
}

void L2Loss::setLabel(cFlowType * & target)
{
    mTarget = *target;
}


double L2Loss::getLoss(void)
{
    return mBatchLoss;
}