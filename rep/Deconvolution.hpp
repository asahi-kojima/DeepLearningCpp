#ifndef ___CLASS_DECONVOLUTION
#define ___CLASS_DECONVOLUTION

#include <iostream>
#include <vector>
#include <random>
#include "BaseLayer.hpp"

class Deconvolution : public BaseLayer{
private:
    S32 mBatchNum;

    S32 mIc;
    S32 mIh;
    S32 mIw;

    S32 mFn;
    S32 mFc;
    S32 mFh;
    S32 mFw;
    S32 mStride;
    S32 mPad;

    S32 mOc;
    S32 mOh;
    S32 mOw;

    S32 mIhIw;
    S32 mIcIhIw;
    S32 mFhFw;
    S32 mFcFhFw;
    S32 mOhOw;
    S32 mOcOhOw;

    S32 mMh;
    S32 mMw;
    S32 mMc;
    S32 mMhMw;
    S32 mMcMhMw;

    ParamType mMatrix;
    ParamType mDMatrix;
    ParamType mBias;
    ParamType mDBias;

    FlowType mReshapedInputData;
    FlowType mForwardResult;
    FlowType mBackwardResult;
    std::vector<std::vector<std::vector<S32> > > mProjectionList;
    std::vector<std::vector<S32> > mProjectionIndex;

    //bool mFlag;

public:
    //******************************
    //       コンストラクタ
    //******************************
    Deconvolution(S32 BatchNum, S32 inputChannel, S32 inputHeight, S32 filterNum, S32 filterHeight, S32 Stride, S32 Pad)
    : mBatchNum(BatchNum)
    , mIc(inputChannel)
    , mIh(inputHeight)
    , mIw(inputHeight)
    , mFn(filterNum)
    , mFc(inputChannel)
    , mFh(filterHeight)
    , mFw(filterHeight)
    , mStride(Stride)
    , mPad(Pad)
    //, mFlag(false)
    , BaseLayer(2)
    {
        mOc = mFn;
        mOh  = (mIh - 1) * mStride + mFh - 2 * mPad;
        mOw  = (mIw - 1) * mStride + mFw - 2 * mPad;
        
        mMh = 2 * (mFh - 1 - mPad) + mIh + (mIh - 1) * mStride;
        mMw = 2 * (mFw - 1 - mPad) + mIw + (mIw - 1) * mStride;//(mIw - 1) * mStride + 2 * mFw - 2 * mPad - 1;
        mMc = mIc;
        mMhMw = mMh * mMw;
        mMcMhMw = mMc * mMhMw;

        mIhIw = mIh * mIw;
        mIcIhIw = mIc * mIhIw;

        mFhFw = mFh * mFw;
        mFcFhFw = mIc * mFhFw;

        mOhOw = mOh * mOw;
        mOcOhOw = mOc * mOhOw;


        mMatrix = ParamType(mFn * mFc * mFh * mFw);
        mDMatrix = ParamType(mFn * mFc * mFh * mFw);
        mBias = ParamType(mOc);
        mDBias = ParamType(mOc);
        std::random_device seed_gen;
        std::default_random_engine engine(seed_gen());
        std::normal_distribution<> dist(0.0, 1.0);
        for (S32 i = 0; i < mOc; i++)
        {
            mBias[i] = 0;
            mDBias[i] = 0;
        }
        for (S32 i = 0; i < mFn * mFh * mFw * mIc; i++)
        {
                mMatrix[i]   = 0.01 * dist(engine);
                mDMatrix[i] = 0;    
        }

        mReshapedInputData = FlowType(mBatchNum, std::vector<F32>(mOhOw * mFcFhFw, 0));
        mForwardResult     = FlowType(mBatchNum, std::vector<F32>(mOcOhOw, 0));
        mBackwardResult    = FlowType(mBatchNum, std::vector<F32>(mIcIhIw, 0));
        mProjectionList = std::vector<std::vector<std::vector<S32> > >(mBatchNum, std::vector<std::vector<S32> >(mIcIhIw, std::vector<S32>(mFhFw, 0)));
        mProjectionIndex = std::vector<std::vector<S32> >(mBatchNum, std::vector<S32>(mIcIhIw, 0));
        
        mEachParamsNum[0] = mFn * mFc * mFh * mFw;
        mEachParamsNum[1] = mOc;
        mParams[0] = &mMatrix;
        mParams[1] = &mBias;
        mDParams[0] = &mDMatrix;
        mDParams[1] = &mDBias;

        calcIndexList();
    };

    
    //******************************
    //       デストラクタ
    //******************************
    ~Deconvolution(){};

    void initialize();
    void forward(cFlowType*&);
    void backward(cFlowType*&);

private:
    void calcIndexList();
};



#endif
