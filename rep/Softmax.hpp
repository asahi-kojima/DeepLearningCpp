#ifndef ___SOFTMAX
#define ___SOFTMAX

#include "Base_Layer.hpp"

class Softmax : public Base_Layer
{
private:
    int m_batchSize;
    int m_dataSize;
    double **m_forwardResult;
    

    
public:
    Softmax(int batchSize, int dataSize)
    : m_batchSize(batchSize)
    , m_dataSize(dataSize)
    {
#ifdef ___ANNOUNCE
        std::cout << "Softmax class constructor started" << std::endl;
#endif 
        m_forwardResult  = new double*[m_batchSize];
        for (int N = 0; N < m_batchSize; N++)
        {
            m_forwardResult[N]  = new double[m_dataSize];
        }

        m_paramsNum = 0;

#ifdef ___ANNOUNCE
        std::cout << "-----> Success!!" << std::endl;
#endif 
    }

    ~Softmax()
    {
#ifdef ___ANNOUNCE
        std::cout << "Softmax class deconstructor started" << std::endl;
#endif

        for (int N = 0; N < m_batchSize; N++)
        {
            delete[] m_forwardResult[N];
        }
        delete[] m_forwardResult;


#ifdef ___ANNOUNCE
        std::cout << "-----> Success!!" << std::endl;
#endif 
    }
    
    void initialize();
    void forward(void *);
    void backward(void *);
};


#endif