#include "Sigmoid.hpp"
#include <iostream>
#include <cmath>

void Sigmoid::initialize(){}

void Sigmoid::forward(void * void_input)
{
    double ** input = (double**)void_input;

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int N = 0; N < m_batchSize; N++)
    {
        for (int i = 0; i < m_inputSize; i++)
        {
            double tmp = 1.0 / (1 + std::exp(-input[N][i]));
            m_forwardResult[N][i] = tmp;
            input[N][i] = tmp;
        }
    }
}

void Sigmoid::backward(void * void_dout)
{
    double ** dout = (double**)void_dout;

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int N = 0; N < m_batchSize; N++)
    {
        for (int i = 0; i < m_inputSize; i++)
        {
            double y = m_forwardResult[N][i];
            dout[N][i] *= y * (1 - y);
        }
    }
}