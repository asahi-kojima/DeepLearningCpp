#ifndef ___CLASS_SOFTMAXWITHLOSS
#define ___CLASS_SOFTMAXWITHLOSS


#include <iostream>
#include <random>
#include <cmath>
#include "Base_Layer.hpp"


/*宣言*/
class SoftmaxWithLoss : public Base_Layer{
private:
    int m_batch_num;
    int m_input_num;
    double **output;
    double *label;
    double batch_loss;

    bool flag;
public:
    //******************************
    //       コンストラクタ
    //******************************
    SoftmaxWithLoss(int batch_num, int input_num)
    :m_batch_num(batch_num)
    ,m_input_num(input_num)
    ,batch_loss(0)
    {
        output = new double*[m_batch_num];
        for (int i = 0; i < m_batch_num; i++)
        {
            output[i] = new double[m_input_num];
        }
        label = new double[m_batch_num];
        flag = false;
    };

    
    //******************************
    //       デコンストラクタ
    //******************************
    ~SoftmaxWithLoss()
    {
        for (int i = 0; i < m_batch_num; i++) 
            delete[] output[i];
        delete[] output;
        delete[] label;
    };


    //******************************
    //       ラベルセッター
    //******************************
    void set_label(double *label_data)
    {
        for (int i = 0; i < m_batch_num; i++)
        {
                label[i] = label_data[i];
        }
    }

    double get_loss(void)
    {
        return batch_loss;
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
    double cross_entropy_error(double **data, double *label_data);


    //******************************
    //       順伝搬関数
    //******************************
    void forward(double** data);

    //******************************
    //       逆伝搬関数
    //******************************
    void backward(double** data);
};



#endif