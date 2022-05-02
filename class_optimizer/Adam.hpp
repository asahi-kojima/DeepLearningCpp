#ifndef ___CLASS_Adam
#define ___CLASS_Adam

#include <iostream>
#include "Base_Optimizer.hpp"
#include "../class_layer/Layer.hpp"

#include <vector>

class Adam : public Base_Optimizer {
public:

    Adam() {
        //std::cout << "Adam class constructor started" << std::endl;
        beta1 = 0.9;
        beta2 = 0.999;
        iter = 0;

        IsReady = std::vector<bool>(30, false);
        m = std::vector<std::vector<double*> >(30,std::vector<double*>(30, nullptr));
        v = std::vector<std::vector<double*> >(30,std::vector<double*>(30, nullptr));
        //std::cout << "Adam class constructor finished" << std::endl;
    }

    virtual ~Adam()
    {
        //std::cout << "Adam class deconstructor started" << std::endl;
        //std::cout << "Adam class deconstructor finished" << std::endl;
    }

    void update(int, Base_Layer*, double);

    double beta1;
    double beta2;
    double iter;

    std::vector<bool> IsReady;
    std::vector<std::vector<double*> > m;
    std::vector<std::vector<double*> > v;

};





#endif