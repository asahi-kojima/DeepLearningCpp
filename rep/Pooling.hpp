#ifndef ___CLASS_POOLING
#define ___CLASS_POOLING



#include <iostream>
#include "Base_Layer.hpp"


/*宣言*/
class Pooling : public Base_Layer
{
private:
    int m_batchNum;
    int m_inputChannel;
    int m_inputHeight;
    int m_inputWidth;

    int m_filterHeight;
    int m_filterWidth;
    int m_stride;
    int m_pad;

    double **m_inputData;
    double **m_forwardResult;
    double **m_backwardResult;

    int m_outputHeight;
    int m_outputWidth;
    int m_outputChannel;

    int **m_resultMemo;

public:
    //******************************
    //       コンストラクタ
    //******************************
    Pooling(int batchNum, int inputChannel, int inputHeight, int inputWidth,
    int filterHeight, int filterWidth, int stride, int pad)
    : m_batchNum(batchNum)
    , m_inputChannel(inputChannel)
    , m_inputHeight(inputHeight)
    , m_inputWidth(inputWidth)
    , m_filterHeight(filterHeight)
    , m_filterWidth(filterWidth)
    , m_stride(stride)
    , m_pad(pad)
    {
        #ifdef ___ANNOUNCE
        std::cout << "Pooling class constructor started" << std::endl;
        #endif 



        m_outputChannel = m_inputChannel;
        m_outputHeight  = 1 + (m_inputHeight - m_filterHeight + 2 * m_pad) / m_stride;
        m_outputWidth   = 1 + (m_inputWidth - m_filterWidth + 2 * m_pad) / m_stride;

        m_inputData         = new double*[m_batchNum];
        m_forwardResult     = new double*[m_batchNum];
        m_backwardResult    = new double*[m_batchNum];
        m_resultMemo = new int*[m_batchNum];
        for (int N = 0; N < m_batchNum; N++)
        {
            //m_reshapedInputData[N] = new double[m_filterHeight * m_filterWidth * m_outputHeight * m_outputWidth * m_outputChannel];
            m_forwardResult[N] = new double[m_outputChannel * m_outputHeight * m_outputWidth];
            m_backwardResult[N] = new double[m_inputChannel * m_inputHeight * m_inputWidth];
            m_resultMemo[N] = new int[m_outputChannel * m_outputHeight * m_outputWidth];
        }


        m_paramsNum = 0;
        
        #ifdef ___ANNOUNCE
        std::cout << "-----> Success!!" << std::endl;
        #endif 
    };

    Pooling(int batchNum, int inputChannel, int inputHeight,
    int filterHeight, int stride, int pad)
    : m_batchNum(batchNum)
    , m_inputChannel(inputChannel)
    , m_inputHeight(inputHeight)
    , m_inputWidth(inputHeight)
    , m_filterHeight(filterHeight)
    , m_filterWidth(filterHeight)
    , m_stride(stride)
    , m_pad(pad)
    {
        #ifdef ___ANNOUNCE
        std::cout << "Pooling class constructor started" << std::endl;
        #endif 



        m_outputChannel = m_inputChannel;
        m_outputHeight  = 1 + (m_inputHeight - m_filterHeight + 2 * m_pad) / m_stride;
        m_outputWidth   = 1 + (m_inputWidth - m_filterWidth + 2 * m_pad) / m_stride;

        m_inputData         = new double*[m_batchNum];
        m_forwardResult     = new double*[m_batchNum];
        m_backwardResult    = new double*[m_batchNum];
        m_resultMemo = new int*[m_batchNum];
        for (int N = 0; N < m_batchNum; N++)
        {
            //m_reshapedInputData[N] = new double[m_filterHeight * m_filterWidth * m_outputHeight * m_outputWidth * m_outputChannel];
            m_forwardResult[N] = new double[m_outputChannel * m_outputHeight * m_outputWidth];
            m_backwardResult[N] = new double[m_inputChannel * m_inputHeight * m_inputWidth];
            m_resultMemo[N] = new int[m_outputChannel * m_outputHeight * m_outputWidth];
        }


        m_paramsNum = 0;
        
        #ifdef ___ANNOUNCE
        std::cout << "-----> Success!!" << std::endl;
        #endif 
    };

    ~Pooling()
    {
        #ifdef ___ANNOUNCE
        std::cout << "Pooling class destructor started" << std::endl;
        #endif 

        for (int N = 0; N < m_batchNum; N++)
        {
            //delete[] m_reshapedInputData[N];
            delete[] m_forwardResult[N];
            delete[] m_backwardResult[N];
            delete[] m_resultMemo[N];
        }

        delete[] m_inputData;
        delete[] m_forwardResult;
        delete[] m_backwardResult;
        delete[] m_resultMemo;

        #ifdef ___ANNOUNCE
        std::cout << "-----> Success!!" << std::endl;
        #endif 
    };

    void initialize();
    void forward(void*);
    void backward(void*);
};



#endif