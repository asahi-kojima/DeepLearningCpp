#include <iostream>
#include "class_layer/Layer.hpp"
//#include "class_optimizer/Optimizer.hpp"
#include "class_dataloader/Mnist.hpp"
#include "class_conductor/Conductor.hpp"

using namespace std;

int main(){
    //**************************************************************************
    //基本情報
    //**************************************************************************
    int data_num = 60000;
    int batch_num = 100;
    int data_size = 28 * 28;

    double lr = 0.001;
    int epochs = 50;

    double **data;
    data = new double*[data_num];
    for (int i = 0; i < data_num; i++) data[i] = new double[data_size];

    double *label;
    label = new double[data_num];
    //**************************************************************************
    //データのロード
    //**************************************************************************
    cout << "data load start" << endl;
    Mnist mnist;
    //データ
    vector<vector<double> > tmp_data;
    tmp_data = mnist.readTrainingFile("class_dataloader/train-images-idx3-ubyte");
    for (int i = 0; i < data_num; i++) for (int j = 0; j < data_size; j++) data[i][j] = tmp_data[i][j];
    //ラベル
    vector<double> tmp_label;
    tmp_label = mnist.readLabelFile("class_dataloader/train-labels-idx1-ubyte");
    for (int i = 0; i < data_num; i++) label[i] = tmp_label[i];

    //**************************************************************************
    //指揮者クラス
    //**************************************************************************
    Conductor *conductor = new Conductor(data_num, batch_num, data_size, lr);
    
    //**************************************************************************
    //学習ループに関する設定
    //**************************************************************************
    conductor->learning(data, label, epochs);
    
    //**************************************************************************
    //検証
    //**************************************************************************
    cout << "検証開始" << endl;
    Mnist mnist2;
    int data_num_for_verify = 10000;
    conductor->get_ptr(data, data_num_for_verify);
    vector<vector<double> > tmp_data2;
    tmp_data2 = mnist.readTrainingFile("class_dataloader/t10k-images-idx3-ubyte");
    for (int i = 0; i < data_num_for_verify; i++){
        for (int j = 0; j < data_size; j++){
            data[i][j] = tmp_data2[i][j];
        }
    }
    vector<double> tmp_label2;
    tmp_label2 = mnist.readLabelFile("class_dataloader/t10k-labels-idx1-ubyte");
    for (int i = 0; i < data_num_for_verify; i++) {
        label[i] = tmp_label2[i];
    }
    conductor->verifying(data, label, data_num_for_verify);
    
    //**************************************************************************
    //クラスの破棄
    //**************************************************************************
    cout << "class delete start" << endl;
    for (int i = 0; i < data_num; i++){
        data[i]  = nullptr;
    }

    delete conductor;



    return 0;
}