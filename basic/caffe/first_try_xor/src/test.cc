// Test a xor net with caffe C++ API
// Author: Yuting Xie
// 2022.3.6

#include <cstdlib>
#include <iostream>
#include <memory>

#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>

#define BATCH_SIZE (64)
#define NUM_BATCH (400)

int main() {
    /*Training*/
    // Generate data
    float datax[BATCH_SIZE * 1 * 1 * 2 * NUM_BATCH];
    float datay[BATCH_SIZE * 1 * 1 * 1 * NUM_BATCH];
    for (int i = 0; i < BATCH_SIZE * 1 * 1 * 1 * NUM_BATCH; i++) {
        datax[i * 2] = rand() % 2;
        datay[i * 2 + 1] = rand() % 2;
        datay[i] = int(datax[i * 2]) ^ int(datax[i * 2 + 1]);
    }

    // Get the caffe solver param from .prototxt
    caffe::SolverParameter solver_param;
    caffe::ReadSolverParamsFromTextFileOrDie("./solver.prototxt", &solver_param);

    // Get a solver from the above params
    std::shared_ptr<caffe::Solver<float>> 
    solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));

    // Obtain the input MemoryData
    caffe::MemoryDataLayer<float> *dataLayer_trainnet = 
    (caffe::MemoryDataLayer<float> *) 
    (solver->net()->layer_by_name("data_input").get());

    // Reset to provide data/labels in memory but the size of each label could only be 1
    dataLayer_trainnet->Reset(datax, datay, BATCH_SIZE * NUM_BATCH);

    // Do train
    solver->Solve();

    return 0;
}
