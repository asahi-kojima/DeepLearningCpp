#ifndef ___CLASS_AFFINE
#define ___CLASS_AFFINE

#include <iostream>
#include <random>
#include <cmath>
#include "Base_Layer.hpp"


class Affine : public Base_Layer
{
public:
    int m_batchNum;
    int m_inputNum;
    int m_outputNum;

    double *m_matrix;
    double *m_dMatrix;
    double *m_bias;
    double *m_dBias;

    double **m_inputData;
    double **m_forwardResult;
    double **m_backwardResult;


public:
    //******************************
    //       コンストラクタ
    //******************************
    Affine(int batchNum, int inputNum, int outputNum)
    :m_batchNum(batchNum)
    ,m_inputNum(inputNum)
    ,m_outputNum(outputNum)
    {
        #ifdef ___ANNOUNCE
        std::cout << "Affine class constructor started" << std::endl;
        #endif   
        //*************************************
        //各種メモリーの確保
        //*************************************
        m_matrix   = new double[m_outputNum * m_inputNum];
        m_dMatrix = new double[m_outputNum * m_inputNum];
        m_bias     = new double[m_outputNum];
        m_dBias   = new double[m_outputNum];
        m_inputData      = new double*[m_batchNum];
        m_forwardResult  = new double*[m_batchNum];
        m_backwardResult = new double*[m_batchNum];

        for (int i = 0; i < m_batchNum; i++)
        {
            m_forwardResult[i]  = new double[m_outputNum];
            m_backwardResult[i] = new double[m_inputNum];
        }
        
        //*************************************
        //バイアスと行列の初期化
        //*************************************
        std::random_device seed_gen;
        std::default_random_engine engine(seed_gen());
        std::normal_distribution<> dist(0.0, 1.0);
        for (int i = 0; i < m_outputNum; i++)
        {
            m_bias[i]   = 0;
            m_dBias[i] = 0;
            for (int j = 0; j < m_inputNum; j++)
            {
                m_matrix[i * m_inputNum + j] = 0.01 * dist(engine) * std::sqrt(2.0 / m_inputNum);
                m_dMatrix[i * m_inputNum + j] = 0.0;    
            }
        }

        //*************************************
        //パラメータのアドレスを格納        
        //*************************************
        m_paramsNum = 2;
        m_eachParamsNum = new int[m_paramsNum];
        m_params   = new double*[m_paramsNum];
        m_dParams = new double*[m_paramsNum];
        
        m_eachParamsNum[0] = m_inputNum * m_outputNum;
        m_eachParamsNum[1] = m_outputNum;

        m_params[0] = m_matrix;
        m_params[1] = m_bias;

        m_dParams[0] = m_dMatrix;
        m_dParams[1] = m_dBias;

        #ifdef ___ANNOUNCE
        std::cout << "-----> Success!!" << std::endl;
        #endif
    };

    
    //******************************
    //       デコンストラクタ
    //******************************
    ~Affine()
    {
        #ifdef ___ANNOUNCE
        std::cout << "Affine class deconstructor started" << std::endl;
        #endif

        //Affine由来の領域解放
        //std::cout << "Affine class deconstructor started" << std::endl;
        for (int i = 0; i < m_batchNum; i++)
        {
            //std::cout << "Affine 1" << std::endl;
            //delete[] m_inputData[i];
            //std::cout << "Affine 2" << std::endl;
            delete[] m_forwardResult[i];
            //std::cout << "Affine 3" << std::endl;
            delete[] m_backwardResult[i];
            //std::cout << "Affine 4" << std::endl;
        }
        //std::cout << "Affine class deconstructor started" << std::endl;
        delete[] m_matrix;
        delete[] m_bias;
        delete[] m_dMatrix;
        delete[] m_dBias;
        delete[] m_inputData;
        delete[] m_forwardResult;
        delete[] m_backwardResult;

        //Base_Layer由来の領域解放
        delete[] m_eachParamsNum;
        delete[] m_params;
        delete[] m_dParams;

        #ifdef ___ANNOUNCE
        std::cout << "-----> Success!!" << std::endl;
        #endif
    };


    //******************************
    //       初期化関数
    //******************************
    void initialize();


    //******************************
    //       順伝搬関数
    //******************************
    void forward(void *);


    //******************************
    //       逆伝搬関数
    //******************************
    void backward(void *);
};





#endif