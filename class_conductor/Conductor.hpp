#ifndef ___CLASS_CONDUCTOR
#define ___CLASS_CONDUCTOR

#include <iostream>
#include <vector>
#include "../class_layer/Layer.hpp"
#include "../class_optimizer/Optimizer.hpp"

class Conductor
{
public:
    int total_data_num;
    int batch_num;
    int each_data_size;
    double lr;

    double **ptr_keeper;

    std::vector<Base_Layer*> layer_list;
    //int layer_num;
    SoftmaxWithLoss *soft;
    Base_Optimizer *Optimizer;

public:
    Conductor(int, int, int, double);

    ~Conductor()
    {
        for (int i = 0; i < layer_list.size(); i++)
        {
            delete layer_list[i];
        }
        delete Optimizer;
    }

    //void setup(int, int, int);
    void get_ptr(double** data, int num)
    {
        for (int i = 0; i < num; i++)
            data[i] = ptr_keeper[i];
    }
    void forward(double**);
    void backward(double**);
    void update();
    void learning(double** , double*, int);
    void verifying(double**, double*, int);
    void data_flatter(double**, int, int);
};




#endif