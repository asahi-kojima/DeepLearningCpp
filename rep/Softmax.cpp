#include "Softmax.hpp"
#include <iostream>
#include <cmath>

void isRangeCorrect2(int n, int min, int max, int ID)
{
    if (!(n >= min && n < max))
    {
        std::cout << "Range Error" << ID << std::endl;
    }
}

void Softmax::initialize(){}

void Softmax::forward(void * void_input)
{
    double ** input = (double**)void_input;

    for (int N = 0; N < m_batchSize; N++)
    {
#ifdef ___ANNOUNCE
        isRangeCorrect2(N, 0, m_batchSize,0);
#endif
        double max = input[N][0];
        for (int i = 0; i < m_dataSize; i++)
        {
#ifdef ___ANNOUNCE
            isRangeCorrect2(i, 0, m_dataSize,1);
#endif
            if (max < input[N][i])
            {
                max = input[N][i];
            }
        }
        double sum = 0;
        for (int i = 0; i < m_dataSize; i++)
        {
            double tmp = std::exp(input[N][i] - max);
            m_forwardResult[N][i] = tmp;
            sum += tmp;
        }
        for (int i = 0; i < m_dataSize; i++)
        {
            m_forwardResult[N][i] /= sum;
            input[N][i] = m_forwardResult[N][i];
            //std::cout << "N = " << N << " | i = " << i << " | data = " << input[N][i] << std::endl;
        }
    }
}

void Softmax::backward(void * void_dout)
{
    double ** dout = (double **)void_dout;


    for (int N = 0; N < m_batchSize; N++)
    {
        int tmp = 0;
        for (int i = 0; i < m_dataSize; i++)
        {
            tmp += dout[N][i] * m_forwardResult[N][i];
        }
        for (int i = 0; i < m_dataSize; i++)
        {
            dout[N][i] = (dout[N][i] - tmp) * m_forwardResult[N][i];
        }
    }
}