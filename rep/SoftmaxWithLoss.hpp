#ifndef ___CLASS_SOFTMAXWITHLOSS
#define ___CLASS_SOFTMAXWITHLOSS


#include <iostream>
#include <random>
#include <cmath>
#include "Base_Layer.hpp"
#include "Base_Output_Layer.hpp"

/*宣言*/
class SoftmaxWithLoss : public Base_Output_Layer
{
private:
    int m_batchNum;
    int m_inputNum;
    double **m_output;
    int *m_label;
    double m_batchLoss;

    bool m_flag;
public:
    //******************************
    //       コンストラクタ
    //******************************
    SoftmaxWithLoss(int batch_num, int input_num)
    :m_batchNum(batch_num)
    ,m_inputNum(input_num)
    ,m_batchLoss(0)
    {
        #ifdef ___ANNOUNCE
        std::cout << "SoftmaxWithLoss class constructor started" << std::endl;
        #endif 

        m_output = new double*[m_batchNum];
        for (int i = 0; i < m_batchNum; i++)
        {
            m_output[i] = new double[m_inputNum];
        }
        m_label = new int[m_batchNum];
        m_flag = false;

        m_paramsNum = 0;

        #ifdef ___ANNOUNCE
        std::cout << "-----> Success!!" << std::endl;
        #endif
    };

    
    //******************************
    //       デコンストラクタ
    //******************************
    ~SoftmaxWithLoss()
    {
        #ifdef ___ANNOUNCE
        std::cout << "SoftmaxWithLoss class deconstructor started" << std::endl;
        #endif
        
        for (int i = 0; i < m_batchNum; i++) 
            delete[] m_output[i];
        delete[] m_output;
        delete[] m_label;

        #ifdef ___ANNOUNCE
        std::cout << "-----> Success!!" << std::endl;
        #endif
    };


    //******************************
    //       ラベルセッター
    //******************************
    void set_label(void * v_m_label_data)
    {
        int * m_label_data = (int *)v_m_label_data;
        for (int i = 0; i < m_batchNum; i++)
        {
            //std::cout <<  i << " | " <<"label = " << m_label_data[i] << std::endl;
            m_label[i] = m_label_data[i];
        }
    }

    double get_loss(void)
    {
        return m_batchLoss;
    }


    //******************************
    //       初期化関数
    //******************************
    void initialize(){}

    //******************************
    //       Softmax関数
    //******************************
    void softmax(double** data);


    //******************************
    //     クロスエントロピー関数
    //******************************
    double cross_entropy_error(double **data, int *m_label_data);


    //******************************
    //       順伝搬関数
    //******************************
    void forward(void *);

    //******************************
    //       逆伝搬関数
    //******************************
    void backward(void *);
};



#endif