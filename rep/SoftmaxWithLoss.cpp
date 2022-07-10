#include "SoftmaxWithLoss.hpp"

#include <unistd.h>
//******************************
//       Softmax関数　実装
//******************************
void SoftmaxWithLoss::softmax(double** data)
{
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < m_batchNum; i++)
    {
        double max = data[i][0];
        for (int j = 0; j < m_inputNum; j++)
        {
            max = (max < data[i][j] ? data[i][j] : max);
        }

        double sum = 0.0;
        for (int j = 0; j < m_inputNum; j++)
        {
            sum += std::exp(data[i][j] - max);
        }
        for (int j = 0; j < m_inputNum; j++)
        {
            data[i][j] = std::exp(data[i][j]-max) / sum;
        }
    }
    
}

//******************************
//     クロスエントロピー　実装
//******************************
double SoftmaxWithLoss::cross_entropy_error(double **data, int *m_label_data)
{
    double sum = 0;
    for (int i = 0; i < m_batchNum; i++)
    {
        double ep = 1 / 100000;
        sum -= std::log(data[i][m_label_data[i]] + ep);
    }
    return sum /= m_batchNum;
}

//******************************
//       順伝搬関数　実装
//******************************
void SoftmaxWithLoss::forward(void * void_input)
{
    double** input = (double**)void_input;
    softmax(input);

#ifdef _OPENMP 
    #pragma omp parallel for
#endif
    for (int i = 0; i < m_batchNum; i++)
    {
        for (int j = 0; j < m_inputNum; j++)
        {
            m_output[i][j] = input[i][j];
        }
    }
    m_batchLoss = cross_entropy_error(input, m_label);
}


//******************************
//       逆伝搬関数　実装
//******************************
void SoftmaxWithLoss::backward(void * void_dout)
{
    double** dout = (double**)void_dout;
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i = 0; i < m_batchNum; i++)
    {
        for (int j = 0; j < m_inputNum; j++)
        {
            dout[i][j] = (m_output[i][j] - (m_label[i] == j ? 1 : 0)) / m_batchNum;
        }
    }


#ifdef ___NUM_CHECK
    for (int i = 0; i < m_batchNum; i++)
    {
        for (int j = 0; j < m_inputNum; j++)
        {
            if (std::isnan(dout[i][j]))
            {
                std::cout << i << " | " << j << std::endl;
                usleep(100 * 1000);
            }
        }
    }
#endif
}
