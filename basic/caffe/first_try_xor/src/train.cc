// Train a xor net with caffe C++ API
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
    /*Testing*/  
    // Initialize a caffe net
    auto test_net = std::make_shared<caffe::Net<float>>("./model.prototxt", caffe::TEST);
    // Load trained layer weights.
    test_net->CopyTrainedLayersFrom("XOR_iter_5000000.caffemodel");

    // Test data
    float testx[] = {0, 0, 0, 1, 1, 0, 1, 1};
    float testy[] = {0, 1, 1, 0};

    // Get input layer (so called blob) from test_net, and feed the test data into it
    auto data_input_layer =
        (caffe::MemoryDataLayer<float>*) (test_net->layer_by_name("test_inputdata").get());
    data_input_layer->Reset(testx, testy, 4); // The batch_size is 4 according to model.prototxt

    // Do test
    test_net->Forward();

    // Obtain network output from the output layer (blob)
    boost::shared_ptr<caffe::Blob<float>> output_layer = 
        test_net->blob_by_name("output");
    const float *result_ptr = output_layer->cpu_data();
    
    std::cout << "Results:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        std::cout << result_ptr[0] << std::endl;
    }

    return 0;
}
