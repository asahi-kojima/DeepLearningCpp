#include <iostream>
#include "class_layer/Layer.hpp"
//#include "class_optimizer/Optimizer.hpp"
#include "class_dataloader/Mnist.hpp"
#include "class_conductor/Conductor.hpp"
//#include "BasicInfo.hpp"

using namespace std;

int main()
{
    //**************************************************************************
    //基本情報
    //**************************************************************************
    int total_data_num;
    int batch_num = 100;
    int each_data_size;

    double lr = 0.001;
    int epochs = 50;

#ifdef _OPENMP
    std::cout << "使用可能な最大スレッド数 : "<< omp_get_max_threads() << std::endl;
#endif

    //**************************************************************************
    // データロード
    //**************************************************************************
    Mnist mnist;
    vector<vector<double> > tmp_data;
    tmp_data = mnist.readTrainingFile("class_dataloader/train-images-idx3-ubyte", &total_data_num, &each_data_size);
    vector<double> tmp_label;
    tmp_label = mnist.readLabelFile("class_dataloader/train-labels-idx1-ubyte");

    double **data;
    data = new double*[total_data_num];
    for (int i = 0; i < total_data_num; i++) data[i] = new double[each_data_size];
    for (int i = 0; i < total_data_num; i++) for (int j = 0; j < each_data_size; j++) data[i][j] = tmp_data[i][j];

    double *label;
    label = new double[total_data_num];
    for (int i = 0; i < total_data_num; i++) label[i] = tmp_label[i];







    //**************************************************************************
    //指揮者クラス
    //**************************************************************************
    Conductor *conductor = new Conductor(total_data_num, batch_num, each_data_size, lr);
    





    //**************************************************************************
    //学習ループに関する設定
    //**************************************************************************
    conductor->learning(data, label, epochs);
    








    //**************************************************************************
    //kesu
    //**************************************************************************
    cout << "検証開始" << endl;
    tmp_data = mnist.readTrainingFile("class_dataloader/train-images-idx3-ubyte", &total_data_num, &each_data_size);
    conductor->get_ptr(data, total_data_num);
    for (int i = 0; i < total_data_num; i++)
    {
        for (int j = 0; j < each_data_size; j++)
        {
            data[i][j] = tmp_data[i][j];
        }
    }
    std::cout << tmp_label[0] << std::endl;
    tmp_label = mnist.readLabelFile("class_dataloader/train-labels-idx1-ubyte");
    std::cout << tmp_label[0] << std::endl;
    for (int i = 0; i < total_data_num; i++)
    {
        label[i] = tmp_label[i];
    }
    conductor->verifying(data, label, total_data_num);




    //**************************************************************************
    //検証
    //**************************************************************************
    cout << "検証開始" << endl;
    tmp_data = mnist.readTrainingFile("class_dataloader/t10k-images-idx3-ubyte", &total_data_num, &each_data_size);
    conductor->get_ptr(data, total_data_num);
    for (int i = 0; i < total_data_num; i++){
        for (int j = 0; j < each_data_size; j++){
            data[i][j] = tmp_data[i][j];
        }
    }
    std::cout << tmp_label[0] << std::endl;
    tmp_label = mnist.readLabelFile("class_dataloader/t10k-labels-idx1-ubyte");
    std::cout << tmp_label[0] << std::endl;
    for (int i = 0; i < total_data_num; i++) {
        label[i] = tmp_label[i];
    }
    conductor->verifying(data, label, total_data_num);
    



    //**************************************************************************
    //クラスの破棄
    //**************************************************************************
    cout << "class delete start" << endl;
    for (int i = 0; i < total_data_num; i++)
    {
        std::cout << i << std::endl;
        delete data[i];
    }
    delete[] data;
    delete label;
    delete conductor;



    return 0;
}