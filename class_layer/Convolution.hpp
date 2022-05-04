#ifndef ___CLASS_CONVOLUTION
#define ___CLASS_CONVOLUTION

#include <iostream>
#include "Base_Layer.hpp"
#include <vector>

class Convolution : public Base_Layer{
private:
    int m_batch_num;
    int m_input_channel;
    int m_input_width;
    int m_input_height;

    int m_filter_num;
    int m_filter_height;
    int m_filter_width;
    int m_stride;
    int m_pad;

    double * matrix;
    double * d_matrix;
    double * bias;
    double * d_bias;

public:
    //******************************
    //       コンストラクタ
    //******************************
    Convolution(int batch_num, int input_channel, int input_witdh, int input_height, int filter_num, int filter_height, int filter_width, int stride, int pad)
    : m_batch_num(batch_num)
    , m_input_channel(input_channel)
    , m_input_width(input_witdh)
    , m_input_height(input_height)
    , m_filter_height(filter_height)
    , m_filter_width(filter_width)
    , m_stride(stride)
    , m_pad(pad)
    {
        if (m_stride <= 0 || pad <= -1)
        {
            std::cout << "Input error" << std::endl;
            abort();
        }

        matrix = new double[m_filter_height * m_filter_width * m_input_channel];
        d_matrix = new double[m_filter_height * m_filter_width * m_input_channel];

        //bias = new double[];
    };

    
    //******************************
    //       デコンストラクタ
    //******************************
    ~Convolution()
    {
        delete matrix;
        delete d_matrix;
    };

    //******************************
    //       初期化関数
    //******************************
    void initialize();


    //******************************
    //       順伝搬関数
    //******************************
    void forward(void *);


    //******************************
    //       逆伝搬関数
    //******************************
    void backward(void *);

private:
};



#endif
