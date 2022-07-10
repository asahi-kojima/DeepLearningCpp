#ifndef ___CLASS_CONVOLUTION
#define ___CLASS_CONVOLUTION

#include <iostream>
#include <vector>
#include <random>
#include "BaseLayer.hpp"

class Convolution : public BaseLayer{
private:
    S32 mBatchSize;
    S32 mInputChannel;
    S32 mInputHeight;
    S32 mInputWidth;

    S32 mFilterNum;
    S32 mFilterHeight;
    S32 mFilterWidth;
    S32 mStride;
    S32 mPad;

    S32 mOutputChannel;
    S32 mOutputHeight;
    S32 mOutputWidth;

    ParamType mMatrix;
    ParamType mDMatrix;
    ParamType mBias;
    ParamType mDBias;

    FlowType mReshapedInputData;
    FlowType mForwardResult;
    FlowType mBackwardResult;


    S32 mIhIw;
    S32 mIcIhIw;
    S32 mIcFhFw;
    S32 mOhOw;
    S32 mOcOhOw;
    S32 mFhFw;

public:
    //******************************
    //       コンストラクタ
    //******************************
    Convolution(S32 batchSize, S32 inputChannel, S32 inputHeight, S32 inputWidth, S32 filterNum, S32 filterHeight, S32 filterWidth, S32 stride, S32 pad)
    : mBatchSize(batchSize)
    , mInputChannel(inputChannel)
    , mInputHeight(inputHeight)
    , mInputWidth(inputWidth)
    , mFilterNum(filterNum)
    , mFilterHeight(filterHeight)
    , mFilterWidth(filterWidth)
    , mStride(stride)
    , mPad(pad)
    , BaseLayer(2)
    {
        mOutputChannel = mFilterNum;
        mOutputHeight  = 1 + (mInputHeight - mFilterHeight + 2 * mPad) / mStride;
        mOutputWidth   = 1 + (mInputWidth  - mFilterWidth  + 2 * mPad) / mStride;

        mIhIw = mInputHeight * mInputWidth;
        mIcIhIw = mInputChannel * mInputHeight * mInputWidth;
        mIcFhFw = mInputChannel * mFilterHeight * mFilterWidth;
        mOhOw = mOutputHeight * mOutputWidth;
        mOcOhOw = mOutputChannel * mOhOw;
        mFhFw = mFilterHeight * mFilterWidth;

        if (mStride <= 0 || mPad <= -1 || mInputWidth < mFilterWidth || mInputHeight < mFilterHeight)
        {
            std::cout << "Input error" << std::endl;
            abort();
        }

        mMatrix  = ParamType(mFilterNum * mIcFhFw);
        mDMatrix = ParamType(mFilterNum * mIcFhFw);
        mBias    = ParamType(mOutputChannel);
        mDBias   = ParamType(mOutputChannel);

        std::random_device seed_gen;
        std::default_random_engine engine(seed_gen());
        std::normal_distribution<> dist(0.0, 1.0);
        for (S32 Out = 0; Out < mOutputChannel; Out++)
        {
            mBias[Out]  = 0;
            mDBias[Out] = 0;
        }
        for (S32 I = 0; I < mFilterNum * mIcFhFw; I++)
        {
                mMatrix[I]   = 0.01 * dist(engine);// / std::sqrt(2.0 / mIcFhFw);
                mDMatrix[I] = 0;
        }

        mReshapedInputData = FlowType(mBatchSize, std::vector<F32>(mOhOw * mIcFhFw, 0));
        mForwardResult     = FlowType(mBatchSize, std::vector<F32>(mOcOhOw, 0));
        mBackwardResult    = FlowType(mBatchSize, std::vector<F32>(mIcIhIw, 0));;

        mEachParamsNum[0] = mFilterNum * mIcFhFw;
        mEachParamsNum[1] = mOutputChannel;
        mParams[0] = &mMatrix;
        mParams[1] = &mBias;

        mDParams[0] = &mDMatrix;
        mDParams[1] = &mDBias;

    };

    //******************************
    //       デコンストラクタ
    //******************************
    ~Convolution()
    {
    };

    //******************************
    //       初期化関数
    //******************************
    void initialize();


    //******************************
    //       順伝搬関数
    //******************************
    void forward(cFlowType*&);


    //******************************
    //       逆伝搬関数
    //******************************
    void backward(cFlowType*&);

};



#endif
