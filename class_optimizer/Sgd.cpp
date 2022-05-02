#include "Sgd.hpp"


void Sgd::update(int, Base_Layer *layer, double lr){
    for (int i = 0; i < (layer->params_num); i++){
        for (int j = 0; j < (layer->each_params_num)[i]; j++){
            layer->params[i][j] -= lr * layer->d_params[i][j];
        }
    }
}