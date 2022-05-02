#ifndef ___CLASS_AFFINE
#define ___CLASS_AFFINE

#include <iostream>
#include <random>
#include <cmath>
#include "Base_Layer.hpp"


class Affine : public Base_Layer
{
public:
    int m_batch_num;
    int m_input_num;
    int m_output_num;

    double *matrix;
    double *d_matrix;
    double *bias;
    double *d_bias;

    double **input_data;
    double **forward_result;
    double **backward_result;


public:
    //******************************
    //       コンストラクタ
    //******************************
    Affine(int batch_num, int input_num, int output_num)
    :m_batch_num(batch_num)
    ,m_input_num(input_num)
    ,m_output_num(output_num)
    {
        //std::cout << "Affine class constructor started" << std::endl;
            
        //*************************************
        //各種メモリーの確保
        //*************************************
        matrix   = new double[m_output_num * m_input_num];
        d_matrix = new double[m_output_num * m_input_num];
        bias     = new double[m_output_num];
        d_bias   = new double[m_output_num];
        input_data      = new double*[m_batch_num];
        forward_result  = new double*[m_batch_num];
        backward_result = new double*[m_batch_num];

        for (int i = 0; i < m_batch_num; i++)
        {
            input_data[i]      = new double[m_input_num];
            forward_result[i]  = new double[m_output_num];
            backward_result[i] = new double[m_input_num];
        }
        
        //*************************************
        //バイアスと行列の初期化
        //*************************************
        std::random_device seed_gen;
        std::default_random_engine engine(seed_gen());
        std::normal_distribution<> dist(0.0, std::sqrt(2.0 / m_input_num));
        for (int i = 0; i < m_output_num; i++)
        {
            bias[i]   = 0;
            d_bias[i] = 0;
            for (int j = 0; j < m_input_num; j++)
            {
                matrix[i * m_input_num + j]   = 0.1 * dist(engine) / std::sqrt(2.0 / m_input_num);
                d_matrix[i * m_input_num + j] = 0.0;    
            }
        }

        //*************************************
        //パラメータのアドレスを格納        
        //*************************************
        params_num = 2;
        each_params_num = new int[params_num];
        params   = new double*[params_num];
        d_params = new double*[params_num];
        
        each_params_num[0] = m_input_num * m_output_num;
        each_params_num[1] = m_batch_num;

        params[0] = matrix;
        params[1] = bias;

        d_params[0] = d_matrix;
        d_params[1] = d_bias;

        std::cout << "Affine class constructor finished" << std::endl;
    };

    
    //******************************
    //       デコンストラクタ
    //******************************
    ~Affine()
    {
        //std::cout << "Affine class deconstructor started" << std::endl;

        //Affine由来の領域解放
        for (int i = 0; i < m_batch_num; i++)
        {
            delete[] input_data[i];
            delete[] forward_result[i];
            delete[] backward_result[i];
        }
        delete   matrix;
        delete   bias;
        delete   d_matrix;
        delete   d_bias;
        delete[] input_data;
        delete[] forward_result;
        delete[] backward_result;

        //Base_Layer由来の領域解放
        delete   each_params_num;
        delete[] params;
        delete[] d_params;

        //std::cout << "Affine class deconstructor finished" << std::endl;
    };


    //******************************
    //       初期化関数
    //******************************
    void initialize();


    //******************************
    //       順伝搬関数
    //******************************
    void forward(double **data);


    //******************************
    //       逆伝搬関数
    //******************************
    void backward(double **data);
};





#endif