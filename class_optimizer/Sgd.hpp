#ifndef ___CLASS_SGD
#define ___CLASS_SGD

#include <iostream>
#include "Base_Optimizer.hpp"
#include "../class_layer/Layer.hpp"


class Sgd : public Base_Optimizer {
public:
    //******************************
    //       コンストラクタ
    //******************************
    Sgd(){
        //std::cout << "SGD class constructor started" << std::endl;
        //std::cout << "SGD class constructor finished" << std::endl;
    }

    //******************************
    //       デコンストラクタ
    //******************************
    virtual ~Sgd(){
        //std::cout << "SGD class deconstructor started" << std::endl;
        //std::cout << "SGD class deconstructor finished" << std::endl;
    }

    //******************************
    //       アップデート関数
    //******************************
    void update(int, Base_Layer*, double);
};





#endif