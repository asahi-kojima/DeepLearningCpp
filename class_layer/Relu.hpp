#ifndef ___CLASS_RELU
#define ___CLASS_RELU

#include <iostream>
#include "Base_Layer.hpp"


/*宣言*/
class Relu : public Base_Layer{
private:
    int m_batch_num;
    int m_input_num;
    int **mask;

public:
    //******************************
    //       コンストラクタ
    //******************************
    Relu(int batch_num, int input_num)
    :m_batch_num(batch_num)
    ,m_input_num(input_num)
    {
        #ifdef ___ANNOUNCE
        std::cout << "ReLU class constructor started" << std::endl;
        #endif 

        params_num = 0;
        mask = new int*[m_batch_num];
        for (int i = 0; i < m_batch_num; i++)
        {
            mask[i] = new int[m_input_num];
            for (int j = 0; j < m_input_num; j++) mask[i][j] = 1;
        }

        #ifdef ___ANNOUNCE
        std::cout << "ReLU class constructor finished" << std::endl;
        #endif 
    };

    ~Relu()
    {
        #ifdef ___ANNOUNCE
        std::cout << "ReLU class deconstructor started" << std::endl;
        #endif 

        for (int i = 0; i < m_batch_num; i++) 
            delete[] mask[i];
        delete[] mask;

        #ifdef ___ANNOUNCE
        std::cout << "ReLU class deconstructor finished" << std::endl;
        #endif 
    };

    void initialize();
    void forward(void*);
    void backward(void*);
};



#endif