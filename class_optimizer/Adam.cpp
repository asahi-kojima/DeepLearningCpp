#include "Adam.hpp"
#include <cmath>

void Adam::update(int order, Base_Layer *layer, double lr)
{
#ifdef _OPENMP
    bool flag = IsReady[order];
    iter++;

    double lr_t = lr * std::sqrt(1.0 - std::pow(beta2, iter)) / (1.0 - std::pow(beta1, iter));
    for (int i = 0; i < (layer->params_num); i++)
    {
        if (!flag)
        {
            m[order][i] = new double[(layer->each_params_num)[i]];
            v[order][i] = new double[(layer->each_params_num)[i]];
            for (int j = 0; j < (layer->each_params_num)[i]; j++)
            {
                m[order][i][j] = 0;
                v[order][i][j] = 0;
            }
        }
        
        for (int j = 0; j < (layer->each_params_num)[i]; j++)
        {
            //layer->params[i][j] -= lr * layer->d_params[i][j];
            double grad  = layer->d_params[i][j];
            double grad2 = std::pow(grad, 2);
            m[order][i][j] += (1 - beta1) * (grad - m[order][i][j]);
            v[order][i][j] += (1 - beta2) * (grad2 - v[order][i][j]);

            layer->params[i][j] -= lr_t * m[order][i][j] / (std::sqrt(v[order][i][j]) + 1e-7);
        }
    }

    if (!flag)
    {
        IsReady[order] = true;
    }
#else
    bool flag = IsReady[order];
    iter++;

    double lr_t = lr * std::sqrt(1.0 - std::pow(beta2, iter)) / (1.0 - std::pow(beta1, iter));
    for (int i = 0; i < (layer->params_num); i++)
    {
        if (!flag)
        {
            m[order][i] = new double[(layer->each_params_num)[i]];
            v[order][i] = new double[(layer->each_params_num)[i]];
            for (int j = 0; j < (layer->each_params_num)[i]; j++)
            {
                m[order][i][j] = 0;
                v[order][i][j] = 0;
            }
        }

        for (int j = 0; j < (layer->each_params_num)[i]; j++)
        {
            //layer->params[i][j] -= lr * layer->d_params[i][j];
            double grad  = layer->d_params[i][j];
            double grad2 = std::pow(grad, 2);
            m[order][i][j] += (1 - beta1) * (grad - m[order][i][j]);
            v[order][i][j] += (1 - beta2) * (grad2 - v[order][i][j]);

            layer->params[i][j] -= lr_t * m[order][i][j] / (std::sqrt(v[order][i][j]) + 1e-7);
        }
    }

    if (!flag)
    {
        IsReady[order] = true;
    }
#endif
}
