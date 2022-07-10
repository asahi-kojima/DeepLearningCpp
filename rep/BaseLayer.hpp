#ifndef ___CLASS_BASE_LAYER
#define ___CLASS_BASE_LAYER

#include <vector>
using F32 = float;
using S32 = int;
using U32 = signed int;

using FlowType = std::vector<std::vector<F32> >;
using cFlowType = const FlowType;
using ParamType = std::vector<F32>;


class BaseLayer
{
public:
    BaseLayer(U32 paramsNum)
    : mParamsNum(paramsNum)
    {
        mParams = std::vector<ParamType*>(paramsNum);
        mDParams = std::vector<ParamType*>(paramsNum);
        mEachParamsNum = std::vector<U32>(paramsNum);
    }
    virtual ~BaseLayer()
    {}
    
    std::vector<ParamType*> mParams;
    std::vector<ParamType*> mDParams;
    U32 mParamsNum;
    std::vector<U32> mEachParamsNum;


    virtual void initialize() = 0;
    virtual void forward(cFlowType*&) = 0;
    virtual void backward(cFlowType*&) = 0;
};

#endif