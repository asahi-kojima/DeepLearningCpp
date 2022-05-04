#ifndef ___CLASS_BASE_LAYER
#define ___CLASS_BASE_LAYER

#include <iostream>
#include "../BasicInfo.hpp"
//#define ___PARALLEL

class Base_Layer
{
public:
    Base_Layer()
    {
        //std::cout << "Base_layer class constructor started" << std::endl;
        //std::cout << "Base layer class constructor finished" << std::endl;
    }
    virtual ~Base_Layer()
    {
        //std::cout << "Base layer class deconstructor started" << std::endl;
        //std::cout << "Base layer class deconstructor finished" << std::endl;
    }
    
    double **params;
    double **d_params;
    int params_num;
    int *each_params_num;


    virtual void initialize() = 0;
    virtual void forward(void *) = 0;
    virtual void backward(void *) = 0;
};

#endif