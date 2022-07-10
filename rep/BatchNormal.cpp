#include <cmath>
#include "BatchNormal.hpp"

void BatchNormal::forward(void * void_input)
{   double ** input = (double **)void_input;
    double ep = 0.00000001;
    //std::cout << "start" << std::endl;
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < m_batchNum; i++)
    {
        double mean = 0;
        for (int j = 0; j < m_dataSize; j++)
        {
            mean += input[i][j];
        }
        mean = mean / m_dataSize;
        double sigma2 = 0;
        for (int j = 0; j < m_dataSize; j++)
        {
            sigma2 += (input[i][j] - mean) * (input[i][j] - mean);
        } 
        m_sigma[i] = std::sqrt(sigma2 + ep);
        for (int j = 0; j < m_dataSize; j++)
        {
            m_output[i][j] = (input[i][j] - mean) / m_sigma[i];
            input[i][j] = m_beta[j] + m_gamma[j] * m_output[i][j];
        }
    }
}


void BatchNormal::backward(void * void_dout)
{
    double ** dout = (double **)void_dout;

#ifdef _OPENMP
    #pragma omp parallel// for schedule(static)
    #pragma omp sections
    {
    #pragma omp section
    #pragma omp parallel for
#endif

    for (int i = 0; i < m_dataSize; i++)
    {
        m_dGamma[i] = 0.0;
        m_dBeta[i] = 0.0;
        double tmp_gamma = 0;
        double tmp_beta = 0;
        for (int N = 0; N < m_batchNum; N++)
        {
            tmp_gamma += m_output[N][i] * dout[N][i];
            tmp_beta += dout[N][i];
        }
        m_dGamma[i] = tmp_gamma;
        m_dBeta[i] = tmp_beta;
    }

#ifdef _OPENMP
    #pragma omp section
    #pragma omp parallel for
#endif

    for (int i = 0; i < m_batchNum; i++)
    {
        double tmp1 = 0.0;
        double tmp2 = 0.0;
        for (int j = 0; j < m_dataSize; j++)
        {
            tmp1 += m_gamma[j] * dout[i][j];
            tmp2 += m_gamma[j] * dout[i][j] * m_output[i][j];
        }
        tmp1 /= m_dataSize;
        for (int j = 0; j < m_dataSize; j++)
        {
            dout[i][j] = (m_gamma[j] * dout[i][j] - tmp1 - m_output[i][j] * tmp2) / m_sigma[i];
        }   
    }

#ifdef _OPENMP
    }
#endif
}