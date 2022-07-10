#include "Affine.hpp"
//#include <omp.h>
#include <unistd.h>

//******************************
//       初期化関数 実装
//******************************
void Affine::initialize()
{
    ;
}


//******************************
//       順伝搬関数　実装
//******************************
void Affine::forward(void * void_input)
{
    double ** input = (double**)(void_input);
#ifdef _OPENMP
    //#pragma omp parallel for// schedule(static)
#endif
    for (int N = 0; N < m_batchNum; N++)
    {
        m_inputData[N] = input[N];

        for (int j = 0; j < m_outputNum; j++)
        {
            double tmp = m_bias[j];
            
            for (int k = 0; k < m_inputNum; k++)
            { 
                tmp += m_matrix[j * m_inputNum + k] * input[N][k];
                
            }
            m_forwardResult[N][j] = tmp;
            
        }
        input[N] = m_forwardResult[N];
    }
}

//******************************
//       逆伝搬関数　実装
//******************************
void Affine::backward(void * void_dout)
{
    double ** dout = (double**)void_dout;
#ifdef _OPENMP
    #pragma omp parallel
    #pragma omp sections
#endif
    {
#ifdef _OPENMP
        #pragma omp section
#endif
        //#pragma omp parallel for num_threads(4)
        for (int i = 0; i < m_outputNum; i++)
        {
            for (int j = 0; j < m_inputNum; j++)
            {
                double tmp = 0;
                for (int k = 0; k < m_batchNum; k++)
                    tmp += m_inputData[k][j] * dout[k][i];
                m_dMatrix[i * m_inputNum + j] = tmp;
            }

            //d_biasの初期化と更新
            m_dBias[i] = 0;
            for (int k = 0; k < m_batchNum; k++) 
            {
                m_dBias[i] += dout[k][i];
            }
        }
        
#ifdef _OPENMP
        #pragma omp section
#endif
        for (int i = 0; i < m_batchNum; i++)
        {
            //std::cout << omp_get_thread_num() << std::endl;
            for (int j = 0; j < m_inputNum; j++)
            {
                double tmp = 0;
                for (int k = 0; k < m_outputNum; k++)
                {
                    tmp += m_matrix[k * m_inputNum + j] * dout[i][k];
                }
                m_backwardResult[i][j] = tmp;
            }

            dout[i] = m_backwardResult[i];
        }
    }
}
