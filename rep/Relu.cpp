#include "Relu.hpp"


//******************************
//       初期化関数
//******************************
void Relu::initialize()
{
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < m_batchNum; i++)
        for (int j = 0; j < m_inputNum; j++)
            m_mask[i][j] = 1;
}

//******************************
//       順伝搬関数　実装
//******************************
void Relu::forward(void * void_input)
{
    double ** input = (double**)void_input;
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < m_batchNum; i++)
    {
        for (int j = 0; j < m_inputNum; j++)
        {
            if (input[i][j] < 0)
            {
                input[i][j] = 0;
                m_mask[i][j] = 0;
            }
        }
    }
}

//******************************
//       逆伝搬関数　実装
//******************************
void Relu::backward(void * void_dout)
{
    double ** dout = (double**)void_dout;
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < m_batchNum; i++)
    {
        for (int j = 0; j < m_inputNum; j++)
            dout[i][j] = dout[i][j] * m_mask[i][j];
    }
}
