#include "Conductor.hpp"

Conductor::Conductor(int total_data_num, int batch_num, int each_data_size, double lr)
:total_data_num(total_data_num)
,batch_num(batch_num)
,each_data_size(each_data_size)
,lr(lr)
{
    Optimizer = new Adam();

    //第一層
    layer_list.push_back(new Affine(     batch_num, each_data_size, 300));
    layer_list.push_back(new BatchNormal(batch_num,                 300));
    layer_list.push_back(new Relu(       batch_num,                 300));
    //第二層
    layer_list.push_back(new Affine(     batch_num, 300           , 100));
    layer_list.push_back(new BatchNormal(batch_num,                 100));
    layer_list.push_back(new Relu(       batch_num,                 100));
    //第三層
    layer_list.push_back(new Affine(     batch_num, 100           , 50));
    layer_list.push_back(new BatchNormal(batch_num,                 50));
    layer_list.push_back(new Relu(       batch_num,                 50));
    //第四層
    layer_list.push_back(new Affine(     batch_num, 50            , 10)); 
    soft = new SoftmaxWithLoss(          batch_num,                 10);
    layer_list.push_back(soft);

    
    ptr_keeper = new double*[total_data_num];
}

void Conductor::forward(double **input)
{
    for (int i = 0; i < layer_list.size(); i++)
    {
        layer_list[i]->initialize();
        layer_list[i]->forward(input);
    }
}

void Conductor::backward(double **d_input)
{
    for (int i = layer_list.size() - 1; i >= 0; i--)
    {
        layer_list[i]->backward(d_input);
    }
}

void Conductor::update()
{
#ifdef _OPENMP
    #pragma omp parallel for
    for (int i = 0; i < layer_list.size(); i++)
    {
        Optimizer->update(i,layer_list[i], lr);
    }
#else
    for (int i = 0; i < layer_list.size(); i++)
    {
        Optimizer->update(i,layer_list[i], lr);
    }
#endif
}

void Conductor::learning(double **data, double *label, int epochs)
{
    data_flatter(data, total_data_num, each_data_size);
    for (int i = 0; i < total_data_num; i++){
        ptr_keeper[i] = data[i];
    }
    int counter = 0;
    double min_loss = 10000.0;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        std::cout << "epoch : " << epoch + 1 << " start" << std::endl;
        for (int iter = 0; iter < (total_data_num / batch_num)/1; iter++)
        {
            if (iter % 10 == 0)
            {
                std::cout << "iter = " << iter << " |  loss = " << soft->get_loss() << std::endl;   
            }
            //ラベルの設定
            soft->set_label(&label[iter * batch_num]);
            //順伝搬
            forward(&data[iter * batch_num]);
            //逆伝搬
            backward(&data[iter * batch_num]);
            //パラメータ更新
            update();
            if (min_loss > soft->get_loss())
            {
                counter = 0;
                min_loss = soft->get_loss();
            } 
            else 
            {
                if (counter < 100)
                    counter++;
                else
                {
                    goto END;
                }
            }
        }
        for (int i = 0; i < total_data_num; i++) data[i] = ptr_keeper[i];
    }
END:
    std::cout << "Forced Termination" << std::endl;
}

void Conductor::data_flatter(double **p, int total_data_num, int each_data_size)
{
    double ep = 0.0000001;
    for (int i = 0; i < total_data_num; i++)
    {
        double mean = 0;
        for (int j = 0; j < each_data_size; j++) mean += p[i][j];
        mean = mean / each_data_size;
        double sigma2 = 0;
        for (int j = 0; j < each_data_size; j++) sigma2 += (p[i][j] - mean) * (p[i][j] - mean);
        double sigma = std::sqrt(sigma2 + ep);
        for (int j = 0; j < each_data_size; j++) p[i][j] = (p[i][j] - mean) / sigma;
    }
}

void Conductor::verifying(double** data, double* label, int total_data_num_for_verify)
{
    data_flatter(data, total_data_num_for_verify, each_data_size);
    double result = 0.0;
    for (int iter = 0; iter < (total_data_num_for_verify / batch_num); iter++)
    {
        for (int i = 0; i < layer_list.size() - 1; i++)
        {
            layer_list[i]->forward(&data[iter * batch_num]);
        }

        for (int i = 0; i < batch_num; i++)
        {
            int index = 0;
            for (int j = 0; j < 10; j++)
            {
                if (data[iter * batch_num + i][index] < data[iter * batch_num + i][j])
                    index = j;
                else
                    ;
            }
            if (index == label[iter * batch_num + i]) result += 1;   
        }
    }
    result = result / total_data_num_for_verify;
    std::cout << "virification result = " << 100 * result << "%" << std::endl;
}