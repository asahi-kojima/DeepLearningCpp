#ifndef ___CLASS_DECONVOLUTION
#define ___CLASS_DECONVOLUTION

#include <iostream>
#include <vector>
#include <random>
#include "Base_Layer.hpp"

class Deconvolution : public Base_Layer{
private:
    int m_batchNum;

    int m_Ic;
    int m_Ih;
    int m_Iw;

    int m_Fn;
    int m_Fc;
    int m_Fh;
    int m_Fw;
    int m_stride;
    int m_pad;

    int m_Oc;
    int m_Oh;
    int m_Ow;

    int m_IhIw;
    int m_IcIhIw;
    int m_FhFw;
    int m_FcFhFw;
    int m_OhOw;
    int m_OcOhOw;

    int m_Mh;
    int m_Mw;
    int m_Mc;
    int m_MhMw;
    int m_McMhMw;

    double * m_matrix;
    double * m_dMatrix;
    double * m_bias;
    double * m_dBias;

    double **m_inputData;
    double **m_reshapedInputData;
    double **m_forwardResult;
    double **m_backwardResult;
    int ***m_projectionList;
    int **m_projectionIndex;

    bool flag;

public:
    //******************************
    //       コンストラクタ
    //******************************
    Deconvolution(int batchNum, int inputChannel, int inputHeight, int filterNum, int filterHeight, int stride, int pad)
    : m_batchNum(batchNum)
    , m_Ic(inputChannel)
    , m_Ih(inputHeight)
    , m_Iw(inputHeight)
    , m_Fn(filterNum)
    , m_Fc(inputChannel)
    , m_Fh(filterHeight)
    , m_Fw(filterHeight)
    , m_stride(stride)
    , m_pad(pad)
    , flag(false)
    {
        #ifdef ___ANNOUNCE
        std::cout << "Convolution class constructor started" << std::endl;
        #endif   
        
        m_Oc = m_Fn;
        m_Oh  = (m_Ih - 1) * m_stride + m_Fh - 2 * m_pad;
        m_Ow  = (m_Iw - 1) * m_stride + m_Fw - 2 * m_pad;
        
        m_Mh = (m_Ih - 1) * m_stride + 2 * m_Fh - 2 * m_pad - 1;
        m_Mw = (m_Iw - 1) * m_stride + 2 * m_Fw - 2 * m_pad - 1;
        m_Mc = m_Ic;
        m_MhMw = m_Mh * m_Mw;
        m_McMhMw = m_Mc * m_MhMw;

        m_IhIw = m_Ih * m_Iw;
        m_IcIhIw = m_Ic * m_IhIw;

        m_FhFw = m_Fh * m_Fw;
        m_FcFhFw = m_Ic * m_FhFw;

        m_OhOw = m_Oh * m_Ow;
        m_OcOhOw = m_Oc * m_OhOw;

        // if (m_stride <= 0 || pad <= -1 || m_inputWidth < m_filterWidth || m_inputHeight < m_filterHeight)
        // {
        //     std::cout << "Input error" << std::endl;
        //     abort();
        // }

        m_matrix = new double[m_Fn * m_Fc * m_Fh * m_Fw];
        m_dMatrix = new double[m_Fn * m_Fc * m_Fh * m_Fw];
        m_bias = new double[m_Oc];
        m_dBias = new double[m_Oc];
        std::random_device seed_gen;
        std::default_random_engine engine(seed_gen());
        std::normal_distribution<> dist(0.0, 1.0);
        for (int i = 0; i < m_Oc; i++)
        {
            m_bias[i] = 0;
            m_dBias[i] = 0;
        }
        for (int i = 0; i < m_Fn * m_Fh * m_Fw * m_Ic; i++)
        {
                m_matrix[i]   = 0.01 * dist(engine);
                m_dMatrix[i] = 0;    
        }

        m_inputData         = new double*[m_batchNum];
        m_reshapedInputData = new double*[m_batchNum];
        m_forwardResult     = new double*[m_batchNum];
        m_backwardResult    = new double*[m_batchNum];
        m_projectionList = new int**[m_batchNum];
        m_projectionIndex = new int*[m_batchNum];
        for (int N = 0; N < m_batchNum; N++)
        {
            m_reshapedInputData[N] = new double[m_OhOw * m_FcFhFw];
            for (int i = 0; i < m_OhOw * m_FcFhFw; i++)
            {
                m_reshapedInputData[N][i] = 0;
            }
            m_forwardResult[N]  = new double[m_OcOhOw];
            m_backwardResult[N] = new double[m_IcIhIw];
            m_projectionList[N] = new int*[m_IcIhIw];
            m_projectionIndex[N] = new int[m_IcIhIw];
            for (int i = 0; i < m_IcIhIw; i++)
            {
                m_projectionIndex[N][i] = 0;
                m_projectionList[N][i] = new int[m_FhFw];
            }
        }

        m_paramsNum = 2;
        m_eachParamsNum = new int[m_paramsNum];
        m_params   = new double*[m_paramsNum];
        m_dParams = new double*[m_paramsNum];
        
        m_eachParamsNum[0] = m_Fn * m_Fc * m_Fh * m_Fw;
        m_eachParamsNum[1] = m_Oc;
        m_params[0] = m_matrix;
        m_params[1] = m_bias;
        m_dParams[0] = m_dMatrix;
        m_dParams[1] = m_dBias;


        

        #ifdef ___ANNOUNCE
        std::cout << "-----> Success!!" << std::endl;
        #endif
    };

    
    //******************************
    //       デストラクタ
    //******************************
    ~Deconvolution()
    {
#ifdef ___ANNOUNCE
        std::cout << "Convolution class destructor started" << std::endl;
#endif


        delete[] m_matrix;
        delete[] m_dMatrix;
        delete[] m_bias;
        delete[] m_dBias;

        for (int i = 0; i < m_batchNum; i++)
        {
            delete[] m_reshapedInputData[i];
            delete[] m_forwardResult[i];
            delete[] m_backwardResult[i];
        }

        delete[] m_inputData;
        delete[] m_reshapedInputData;
        delete[] m_forwardResult;
        delete[] m_backwardResult;


        #ifdef ___ANNOUNCE
        std::cout << "-----> Success!!" << std::endl;
        #endif
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
