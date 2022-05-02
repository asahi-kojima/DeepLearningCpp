#ifndef ___CLASS_BASE_OPTIMIZER
#define ___CLASS_BASE_OPTIMIZER
#include "../class_layer/Layer.hpp"

class Base_Optimizer{
public:
    Base_Optimizer(){
        //std::cout << "Base Optimizer class constructor started" << std::endl;
        //std::cout << "Base Optimizer class constructor finished" << std::endl;
    }
    virtual ~Base_Optimizer(){
        //std::cout << "Base Optimizer class deconstructor started" << std::endl;
        //std::cout << "Base Optimizer class deconstructor finished" << std::endl;
    }
    virtual void update(int, Base_Layer*, double) = 0;
};

#endif