#include "SoftmaxWithLoss.hpp"


//******************************
//       Softmax関数　実装
//******************************
void SoftmaxWithLoss::softmax(double** data)
{
    for (int i = 0; i < m_batch_num; i++)
    {
        double max = data[i][0];
        for (int j = 0; j < m_input_num; j++)
        {
            max = (max < data[i][j] ? data[i][j] : max);
        }
        double sum = 0.0;
        for (int j = 0; j < m_input_num; j++)
        {
            sum += std::exp(data[i][j] - max);
        }
        for (int j = 0; j < m_input_num; j++)
        {
            data[i][j] = std::exp(data[i][j]-max) / sum;
        }
    }
}

//******************************
//     クロスエントロピー　実装
//******************************
double SoftmaxWithLoss::cross_entropy_error(double **data, double *label_data)
{
    double sum = 0;
    for (int i = 0; i < m_batch_num; i++)
    {
        double ep = 1 / 100000;
        sum -= std::log(data[i][(int)label_data[i]] + ep);
    }
    return sum /= m_batch_num;
}

//******************************
//       順伝搬関数　実装
//******************************
void SoftmaxWithLoss::forward(void * v_data)
{
    double** data = (double**)v_data;
    softmax(data);
    for (int i = 0; i < m_batch_num; i++)
    {
        for (int j = 0; j < m_input_num; j++)
        {
            output[i][j] = data[i][j];
        }
        
    }
    batch_loss = cross_entropy_error(data, label);
}


//******************************
//       逆伝搬関数　実装
//******************************
void SoftmaxWithLoss::backward(void * v_data)
{
    double** data = (double**)v_data;
    for (int i = 0; i < m_batch_num; i++)
    {
        for (int j = 0; j < m_input_num; j++)
        {
            data[i][j] = (output[i][j] - (label[i] == j ? 1 : 0)) / m_batch_num;
        }
    }
}
