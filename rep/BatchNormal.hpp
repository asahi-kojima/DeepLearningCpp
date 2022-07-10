#ifndef ___CLASS_BATCH_NORMAL
#define ___CLASS_BATCH_NORMAL

#include "Base_Layer.hpp"

class BatchNormal : public Base_Layer
{
public:
    int m_batchNum;
    int m_dataSize;
    double **m_output;
    double *m_sigma;

    double *m_gamma;
    double *m_beta;
    double *m_dGamma;
    double *m_dBeta;

    BatchNormal(int batchNum, int dataSize)
    : m_batchNum(batchNum)
    , m_dataSize(dataSize)
    {
        #ifdef ___ANNOUNCE
        std::cout << "BatchNormal class constructor started" << std::endl;
        #endif 
        

        m_output = new double*[m_batchNum];
        for (int i = 0; i < m_batchNum; i++) m_output[i] = new double[m_dataSize];
        m_sigma = new double[m_batchNum];

        m_gamma   = new double[m_dataSize];
        m_dGamma = new double[m_dataSize];
        m_beta    = new double[m_dataSize];
        m_dBeta  = new double[m_dataSize];

        for (int i = 0; i < m_dataSize; i++)
        {
            m_gamma[i] = 1.0;
            m_dGamma[i] = 0.0;
            m_beta[i] = 0.0;
            m_dBeta[i] = 0.0; 
        }


        m_paramsNum = 2;
        m_eachParamsNum = new int[m_paramsNum];
        m_params   = new double*[m_paramsNum];
        m_dParams = new double*[m_paramsNum];

        m_eachParamsNum[0] = m_dataSize;
        m_eachParamsNum[1] = m_dataSize;

        m_params[0] = m_gamma;
        m_params[1] = m_beta;

        m_dParams[0] = m_dGamma;
        m_dParams[1] = m_dBeta;

        #ifdef ___ANNOUNCE
        std::cout << "-----> Success!!" << std::endl;
        #endif 
    }
    ~BatchNormal()
    {
        #ifdef ___ANNOUNCE
        std::cout << "BatchNormal class deconstructor started" << std::endl;
        #endif 

        for (int i = 0; i < m_batchNum; i++) delete[] m_output[i];
        delete[] m_output;
        delete[] m_sigma;

        delete[] m_gamma;
        delete[] m_beta;
        delete[] m_dGamma;
        delete[] m_dBeta;

        #ifdef ___ANNOUNCE
        std::cout << "-----> Success!!" << std::endl;
        #endif 
    }

    void initialize(){}
    void forward(void *);
    void backward(void *);
};



#endif