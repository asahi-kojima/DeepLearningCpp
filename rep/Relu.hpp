#ifndef ___CLASS_RELU
#define ___CLASS_RELU

#include <iostream>
#include "Base_Layer.hpp"


/*宣言*/
class Relu : public Base_Layer
{
private:
    int m_batchNum;
    int m_inputNum;
    int **m_mask;

public:
    //******************************
    //       コンストラクタ
    //******************************
    Relu(int batch_num, int input_num)
    :m_batchNum(batch_num)
    ,m_inputNum(input_num)
    {
        #ifdef ___ANNOUNCE
        std::cout << "ReLU class constructor started" << std::endl;
        #endif 

        m_paramsNum = 0;
        m_mask = new int*[m_batchNum];
        for (int i = 0; i < m_batchNum; i++)
        {
            m_mask[i] = new int[m_inputNum];
            for (int j = 0; j < m_inputNum; j++)
            {
                m_mask[i][j] = 1;
            }
        }

        #ifdef ___ANNOUNCE
        std::cout << "-----> Success!!" << std::endl;
        #endif 
    };

    ~Relu()
    {
        #ifdef ___ANNOUNCE
        std::cout << "ReLU class deconstructor started" << std::endl;
        #endif 

        for (int i = 0; i < m_batchNum; i++) 
            delete[] m_mask[i];
        delete[] m_mask;

        #ifdef ___ANNOUNCE
        std::cout << "-----> Success!!" << std::endl;
        #endif 
    };

    void initialize();
    void forward(void*);
    void backward(void*);
};



#endif