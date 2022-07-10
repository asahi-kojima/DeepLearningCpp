#ifndef ___L2LOSS
#define ___L2LOSS

#include "BaseOutputLayer.hpp"

class L2Loss : public BaseOutputLayer
{
private:
    U32 mBatchSize;
    U32 mDataSize;
    U32 mOutputSize;
    FlowType mForwardResult;
    FlowType mBackwardResult;

    FlowType mIntermediumResult;
    FlowType mTarget;
    F32 mBatchLoss;
    
public:
    L2Loss(int batchSize, int dataSize)
    : mBatchSize(batchSize)
    , mDataSize(dataSize)
    , BaseOutputLayer(0)
    {
        mForwardResult  = FlowType(mBatchSize, std::vector<F32>(1, 0));
        mBackwardResult = FlowType(mBatchSize, std::vector<F32>(mDataSize, 0));
        mBackwardResult = FlowType(mBatchSize, std::vector<F32>(mDataSize, 0));
    }


    ~L2Loss(){}


    virtual void initialize();
    virtual void forward(cParamType*&);
    virtual void backward(cParamType*&);
    virtual void setLabel(void *);
    virtual double getLoss(void);
};



#endif