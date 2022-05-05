#ifndef ___CLASS_BASE_OUTPUT_LAYER
#define ___CLASS_BASE_OUTPUT_LAYER

#include <iostream>
#include "../BasicInfo.hpp"
#include "Base_Layer.hpp"
//#define ___PARALLEL

class Base_Output_Layer : public Base_Layer
{
public:
    Base_Output_Layer()
    {
        //std::cout << "Base_layer class constructor started" << std::endl;
        //std::cout << "Base layer class constructor finished" << std::endl;
    }
    virtual ~Base_Output_Layer()
    {
        //std::cout << "Base layer class deconstructor started" << std::endl;
        //std::cout << "Base layer class deconstructor finished" << std::endl;
    }

    virtual void set_label(void *) = 0;
    virtual double get_loss(void) = 0;
};

#endif