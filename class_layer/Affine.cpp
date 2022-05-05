#include "Affine.hpp"
//#include <omp.h>

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
void Affine::forward(void * v_input)
{
    double ** input = (double**)(v_input);

#ifdef _OPENMP
    #pragma omp parallel for// schedule(static)
#endif
    for (int i = 0; i < m_batch_num; i++)
    {
        input_data[i] = std::move(input[i]);

        for (int j = 0; j < m_output_num; j++)
        {
            double tmp = bias[j];
            for (int k = 0; k < m_input_num; k++)
                tmp += matrix[j * m_input_num + k] * input[i][k];
            forward_result[i][j] = tmp;
        }

        input[i] = forward_result[i];
    }
}

//******************************
//       逆伝搬関数　実装
//******************************
void Affine::backward(void * v_dout)
{
    double ** dout = (double**)v_dout;
#ifdef _OPENMP
    #pragma omp parallel
    #pragma omp sections
#endif
    {
#ifdef _OPENMP
        #pragma omp section
#endif
        //#pragma omp parallel for num_threads(4)
        for (int i = 0; i < m_output_num; i++)
        {
            //std::cout << omp_get_thread_num() << std::endl;
            //d_matrixの初期化と更新
            for (int j = 0; j < m_input_num; j++)
            {
                double tmp = 0;
                for (int k = 0; k < m_batch_num; k++)
                    tmp += input_data[k][j] * dout[k][i];
                d_matrix[i * m_input_num + j] = tmp;
            }

            //d_biasの初期化と更新
            d_bias[i] = 0;
            for (int k = 0; k < m_batch_num; k++) d_bias[i] += dout[k][i];
        }
        
#ifdef _OPENMP
        #pragma omp section
#endif
        for (int i = 0; i < m_batch_num; i++)
        {
            //std::cout << omp_get_thread_num() << std::endl;
            for (int j = 0; j < m_input_num; j++)
            {
                double tmp = 0;
                for (int k = 0; k < m_output_num; k++)
                {
                    tmp += matrix[k * m_input_num + j] * dout[i][k];
                }
                backward_result[i][j] = tmp;
            }

            dout[i] = backward_result[i];
        }
    }
}
