#ifndef ___CLASS_CONVOLUTION
#define ___CLASS_CONVOLUTION

#include <iostream>
#include "Base_Layer.hpp"
#include <vector>

class Convolution : public Base_Layer{
private:
    int m_batch_num;
    int m_input_width;
    int m_input_height;
    int m_stride;
    int m_pad;

public:
    //******************************
    //       コンストラクタ
    //******************************
    Convolution(int batch_num, int input_witdh, int input_height, int stride, int pad)
    : m_batch_num(batch_num)
    , m_input_width(input_witdh)
    , m_input_height(input_height)
    , m_stride(stride)
    , m_pad(pad)
    {
           
    };

    
    //******************************
    //       デコンストラクタ
    //******************************
    ~Convolution()
    {
    
    };

    //******************************
    //       初期化関数
    //******************************
    void initialize();


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
