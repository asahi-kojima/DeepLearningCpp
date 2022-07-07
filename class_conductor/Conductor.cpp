#include "Conductor.hpp"
#include <iomanip>
Conductor::Conductor(int TotalDataNum, int BatchNum, int EachDataSize, double lr)
:mTotalDataNum(TotalDataNum)
,mBatchNum(BatchNum)
,mEachDataSize(EachDataSize)
,lr(lr)
{
    #ifdef ___ANNOUNCE
    std::cout << std::endl << std::endl;
    std::cout << "--------- Conductor class constructor started ----------" << std::endl;
    #endif

    std::cout << "オプティマイザー" << std::endl;
    Optimizer = new Adam();

    int ID = 1;
    //第一層
    std::cout << "第" << ID++ << "層" << std::endl;
    layer_list.push_back(new Convolution(mBatchNum, 5, mEachDataSize, 256, 4, 2, 1));
    layer_list.push_back(new Relu       (mBatchNum, 256 * (mEachDataSize / 2) * (mEachDataSize / 2)));
    layer_list.push_back(new BatchNormal(mBatchNum, 256 * (mEachDataSize / 2) * (mEachDataSize / 2)));
    //第二層
    std::cout << "第" << ID++ << "層" << std::endl;
    layer_list.push_back(new Convolution(mBatchNum, 256, mEachDataSize / 2, 512, 4, 2, 1));
    layer_list.push_back(new Relu       (mBatchNum, 16 * (mEachDataSize / 4) * (mEachDataSize / 4)));
    layer_list.push_back(new BatchNormal(mBatchNum, 16 * (mEachDataSize / 4) * (mEachDataSize / 4)));
    //第三層
    std::cout << "第" << ID++ << "層" << std::endl;
    layer_list.push_back(new Deconvolution(mBatchNum, 512, mEachDataSize / 4, 256, 4, 2, 1));
    layer_list.push_back(new Relu         (mBatchNum, 256 * (mEachDataSize / 2) * (mEachDataSize / 2)));
    layer_list.push_back(new BatchNormal  (mBatchNum, 256 * (mEachDataSize / 2) * (mEachDataSize / 2)));
    //第四層
    std::cout << "第" << ID++ << "層" << std::endl;
    layer_list.push_back(new Deconvolution(mBatchNum, 256, mEachDataSize / 2, 128, 4, 2, 1));
    layer_list.push_back(new Relu         (mBatchNum, 128 * mEachDataSize * mEachDataSize));
    layer_list.push_back(new BatchNormal  (mBatchNum, 128 * mEachDataSize * mEachDataSize));
    //第五層
    std::cout << "第" << ID++ << "層" << std::endl;
    layer_list.push_back(new Deconvolution(mBatchNum, 128, mEachDataSize, 64, 4, 2, 1));
    layer_list.push_back(new Relu         (mBatchNum, 64 * (2 * mEachDataSize) * (2 * mEachDataSize)));
    layer_list.push_back(new BatchNormal  (mBatchNum, 64 * (2 * mEachDataSize) * (2 * mEachDataSize)));
    //第五層
    std::cout << "第" << ID++ << "層" << std::endl;
    layer_list.push_back(new Deconvolution(mBatchNum, 64, 2 * mEachDataSize, 5, 4, 2, 1));

    output_layer = new L2Loss(mBatchNum, 4 * mEachDataSize);
    layer_list.push_back(output_layer);
    

    originalData = new double*[mTotalDataNum];

    #ifdef ___ANNOUNCE
    std::cout << "--------- Conductor class constructor Success ----------" << std::endl;
    std::cout << std::endl << std::endl;
    #endif
}

void Conductor::forward(double **input)
{
    for (int i = 0; i < layer_list.size(); i++)
    {
        layer_list[i]->initialize();
        layer_list[i]->forward(input);
    }
}

void Conductor::backward(double **dout)
{
    for (int i = layer_list.size() - 1; i >= 0; i--)
    {
        layer_list[i]->backward(dout);
    }
}

void Conductor::update()
{
    for (int i = 0; i < layer_list.size(); i++)
    {
        Optimizer->update(i,layer_list[i], lr);
    }
}

void Conductor::learning(void * v_data, void * v_label, int epochs)
{
    double ** data = (double **)v_data;
    int * label = (int*)v_label;

    dataFlatter(data, mTotalDataNum, mEachDataSize);

    for (int i = 0; i < mTotalDataNum; i++) 
    {
        originalData[i] = data[i];
    }
    int counter = 0;
    double min_loss = 10000.0;
    int startPoint = 0;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        std::cout << "<<<< epoch : " << epoch + 1 << " start >>>>" << std::endl;
        for (int iter = 0; iter < (mTotalDataNum / mBatchNum); iter++)
        {
            if ((iter+1) % 10 == 0)
            {
                std::cout << "iter = " << std::setw(4) << iter + 1 << " |  loss = " << std::setprecision(3) <<output_layer->get_loss() << std::endl;   
            }
            
            startPoint = iter * mBatchNum;
            //ラベルの設定
            output_layer->set_label(&(originalData[startPoint]));
            //順伝搬
            forward(&(data[startPoint]));
            //逆伝搬
            backward(&(data[startPoint]));
            //パラメータ更新
            update();

            if (min_loss > output_layer->get_loss())
            {
                counter = 0;
                min_loss = output_layer->get_loss();
            } 
            else 
            {
                if (counter < 100 || epoch < 3)
                    counter++;
                else
                {
                    goto END;
                }
            }
        }
        for (int i = 0; i < mTotalDataNum; i++) 
        {
            data[i] = originalData[i];
        }
        std::cout << std::endl;
    }
END:
    std::cout << std::endl << std::endl << std::endl;
    std::cout << "<<< Normal Termination >>>" << std::endl;
    std::cout << "Learning was forced to terminate" << std::endl;
    std::cout << "because the decrease in the loss function reached its peak." << std::endl;
    std::cout << "This is normal termination." << std::endl;
    std::cout << std::endl << std::endl << std::endl;
    for (int i = 0; i < mTotalDataNum; i++) data[i] = originalData[i];
}

void Conductor::dataFlatter(double **p, int mTotalDataNum, int mEachDataSize)
{
    double ep = 1e-7;
    for (int i = 0; i < mTotalDataNum; i++)
    {
        double mean = 0;
        for (int j = 0; j < mEachDataSize; j++) mean += p[i][j];
        mean = mean / mEachDataSize;
        double sigma2 = 0;
        for (int j = 0; j < mEachDataSize; j++) sigma2 += (p[i][j] - mean) * (p[i][j] - mean);
        double sigma = std::sqrt(sigma2 + ep);
        for (int j = 0; j < mEachDataSize; j++) p[i][j] = (p[i][j] - mean) / sigma;
    }
}

void Conductor::verification(void * v_data, void * v_label, int total_data_num_for_verify)
{
    double ** data = (double **)v_data; 
    int * label = (int *)v_label;

    double ** p = new double*[total_data_num_for_verify];
    for (int i = 0; i < total_data_num_for_verify; i++) p[i] = data[i];

    dataFlatter(data, total_data_num_for_verify, mEachDataSize);
    double result = 0.0;

    for (int iter = 0; iter < (total_data_num_for_verify / mBatchNum); iter++)
    {
        for (int i = 0; i < layer_list.size() - 1; i++)
        {
            layer_list[i]->forward(&data[iter * mBatchNum]);
        }

        for (int i = 0; i < mBatchNum; i++)
        {
            int index = 0;
            for (int j = 0; j < 10; j++)
            {
                if (data[iter * mBatchNum + i][index] < data[iter * mBatchNum + i][j])
                    index = j;
                else
                    ;
            }
            if (index == label[iter * mBatchNum + i]) result += 1;   
        }
    }

    for (int i = 0; i < total_data_num_for_verify; i++) data[i] = p[i];
    delete[] p;

    result = result / total_data_num_for_verify;
    std::cout << "virification result = " << 100 * result << "%" << std::endl;
}