#ifndef ___CLASS_BASE_OUTPUT_LAYER
#define ___CLASS_BASE_OUTPUT_LAYER

#include "BaseLayer.hpp"
//#define ___PARALLEL

class BaseOutputLayer : public BaseLayer
{
public:
    BaseOutputLayer(U32 paramsNum):Base_Layer(paramsNum){}
    virtual ~BaseOutputLayer(){}

    virtual void setLabel(cFlowType * &) = 0;
    virtual F32 getLoss(void) = 0;
};

#endif