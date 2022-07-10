#ifndef ___SIGMOID
#define ___SIGMOID

#include "Base_Layer.hpp"

class Sigmoid : public Base_Layer
{
public:
    int m_batchSize;
    int m_inputSize;
    double ** m_forwardResult;

    Sigmoid(int batchSize, int inputSize)
    : m_batchSize(batchSize)
    , m_inputSize(inputSize)
    {
#ifdef ___ANNOUNCE
        std::cout << "Sigmoid class constructor started" << std::endl;
#endif 

    m_paramsNum = 0;

    m_forwardResult = new double*[m_batchSize];
    for(int N = 0; N < m_batchSize; N++)
    {
        m_forwardResult[N] = new double[m_inputSize];
    }

#ifdef ___ANNOUNCE
        std::cout << "-----> Success!!" << std::endl;
#endif 
    }

    ~Sigmoid()
    {
#ifdef ___ANNOUNCE
        std::cout << "Sigmoid class destructor started" << std::endl;
#endif 

        for(int N = 0; N < m_batchSize; N++)
        {
            delete[] m_forwardResult[N];
        }
        delete[] m_forwardResult;

#ifdef ___ANNOUNCE
        std::cout << "-----> Success!!" << std::endl;
#endif 
    }

    void initialize();
    void forward(void*);
    void backward(void*);


};

#endif