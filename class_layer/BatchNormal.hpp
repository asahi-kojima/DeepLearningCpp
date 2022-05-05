#ifndef ___CLASS_BATCH_NORMAL
#define ___CLASS_BATCH_NORMAL

#include "Base_Layer.hpp"

class BatchNormal : public Base_Layer
{
public:
    int batch_num;
    int data_size;
    double **output;
    double *sigma;

    double *gamma;
    double *beta;
    double *d_gamma;
    double *d_beta;

    BatchNormal(int batch_num, int data_size)
    {
        #ifdef ___ANNOUNCE
        std::cout << "BatchNormal class constructor started";
        #endif 

        this->batch_num = batch_num;
        this->data_size = data_size;
        

        output = new double*[batch_num];
        for (int i = 0; i < batch_num; i++) output[i] = new double[data_size];
        sigma = new double[batch_num];

        gamma   = new double[data_size];
        d_gamma = new double[data_size];
        beta    = new double[data_size];
        d_beta  = new double[data_size];

        for (int i = 0; i < data_size; i++)
        {
            gamma[i] = 1.0;
            d_gamma[i] = 0.0;
            beta[i] = 0.0;
            d_beta[i] = 0.0; 
        }


        params_num = 2;
        each_params_num = new int[params_num];
        params   = new double*[params_num];
        d_params = new double*[params_num];

        each_params_num[0] = data_size;
        each_params_num[1] = data_size;

        params[0] = gamma;
        params[1] = beta;

        d_params[0] = d_gamma;
        d_params[1] = d_beta;

        #ifdef ___ANNOUNCE
        std::cout << "-----> Success!!" << std::endl;
        #endif 
    }
    ~BatchNormal()
    {
        #ifdef ___ANNOUNCE
        std::cout << "BatchNormal class deconstructor started";
        #endif 

        for (int i = 0; i < batch_num; i++) delete[] output[i];
        delete[] output;
        delete[] sigma;

        #ifdef ___ANNOUNCE
        std::cout << "-----> Success!!" << std::endl;
        #endif 
    }

    void initialize(){}
    void forward(void *);
    void backward(void *);
};



#endif