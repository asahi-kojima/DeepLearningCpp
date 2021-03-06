#include "Relu.hpp"


//******************************
//       初期化関数
//******************************
void Relu::initialize()
{
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < m_batch_num; i++)
        for (int j = 0; j < m_input_num; j++)
            mask[i][j] = 1;
}

//******************************
//       順伝搬関数　実装
//******************************
void Relu::forward(void * v_data)
{
    double ** data = (double**)v_data;
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < m_batch_num; i++)
    {
        for (int j = 0; j < m_input_num; j++)
        {
            if (data[i][j] < 0)
            {
                data[i][j] = 0;
                mask[i][j] = 0;
            }
        }
    }
}

//******************************
//       逆伝搬関数　実装
//******************************
void Relu::backward(void * v_data)
{
    double ** data = (double**)v_data;
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < m_batch_num; i++)
    {
        for (int j = 0; j < m_input_num; j++)
            data[i][j] = data[i][j] * mask[i][j];
    }
}
