#include <cmath>
#include "BatchNormal.hpp"

void BatchNormal::forward(double **data){
    double ep = 0.00000001;
#ifdef _OPENMP
    #pragma omp parallel for
        for (int i = 0; i < batch_num; i++){
            double mean = 0;
            for (int j = 0; j < data_size; j++) 
                mean += data[i][j];
            mean = mean / data_size;
            double sigma2 = 0;
            for (int j = 0; j < data_size; j++) 
                sigma2 += (data[i][j] - mean) * (data[i][j] - mean);
            sigma[i] = std::sqrt(sigma2 + ep);
            for (int j = 0; j < data_size; j++)
            {
                output[i][j] = (data[i][j] - mean) / sigma[i];
                data[i][j] = beta[j] + gamma[j] * output[i][j];
            }
        }
#else
    for (int i = 0; i < batch_num; i++){
        double mean = 0;
        for (int j = 0; j < data_size; j++) 
            mean += data[i][j];
        mean = mean / data_size;
        double sigma2 = 0;
        for (int j = 0; j < data_size; j++) 
            sigma2 += (data[i][j] - mean) * (data[i][j] - mean);
        sigma[i] = std::sqrt(sigma2 + ep);
        for (int j = 0; j < data_size; j++)
        {
            output[i][j] = (data[i][j] - mean) / sigma[i];
            data[i][j] = beta[j] + gamma[j] * output[i][j];
        }
    }
#endif
}


void BatchNormal::backward(double **d_data)
{
#ifdef _OPENMP
    #pragma omp parallel// for schedule(static)
    #pragma omp sections
    {
        #pragma omp section
        for (int i = 0; i < data_size; i++)
        {
            d_gamma[i] = 0.0;
            d_beta[i] = 0.0;
            double tmp_gamma = 0;
            double tmp_beta = 0;
            for (int n = 0; n < batch_num; n++)
            {
                tmp_gamma += output[n][i] * d_data[n][i];
                tmp_beta += d_data[n][i];
            }
            d_gamma[i] = tmp_gamma;
            d_beta[i] = tmp_beta;
        }

        #pragma omp section
        for (int i = 0; i < batch_num; i++)
        {
            double tmp1 = 0.0;
            double tmp2 = 0.0;
            for (int j = 0; j < data_size; j++)
            {
                tmp1 += gamma[j] * d_data[i][j];
                tmp2 += gamma[j] * d_data[i][j] * output[i][j];
            }
            tmp1 /= data_size;
            for (int j = 0; j < data_size; j++)
                d_data[i][j] = (gamma[j] * d_data[i][j] - tmp1 - output[i][j] * tmp2) / sigma[i];
        }
    }
#else
    for (int i = 0; i < data_size; i++)
    {
        d_gamma[i] = 0.0;
        d_beta[i] = 0.0;
        double tmp_gamma = 0;
        double tmp_beta = 0;
        for (int n = 0; n < batch_num; n++)
        {
            tmp_gamma += output[n][i] * d_data[n][i];
            tmp_beta += d_data[n][i];
        }
        d_gamma[i] = tmp_gamma;
        d_beta[i] = tmp_beta;
    }

    for (int i = 0; i < batch_num; i++)
    {
        double tmp1 = 0.0;
        double tmp2 = 0.0;
        for (int j = 0; j < data_size; j++)
        {
            tmp1 += gamma[j] * d_data[i][j];
            tmp2 += gamma[j] * d_data[i][j] * output[i][j];
        }
        tmp1 /= data_size;
        for (int j = 0; j < data_size; j++)
            d_data[i][j] = (gamma[j] * d_data[i][j] - tmp1 - output[i][j] * tmp2) / sigma[i];
    }
#endif
}