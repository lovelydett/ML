// Infer with a trained xor net with libtorch C++ API
// Author: Yuting Xie
// 2022.3.7

#include <torch/script.h>

#include <iostream>
#include <vector>
#include <memory>

int main(){
    // Model
    torch::jit::Module net;

    // Load model
    try {
        net = torch::jit::load("../model/xor_model.pt");
    }
    catch(...) {
        std::cerr << "Unable to load network model\n";
        return -1;
    }

    // Get one piece of input
    std::vector<torch::jit::IValue> one_input;

    // Fill this ONE input (only emplace_back once to fill a how line)
    one_input.emplace_back(torch::tensor({1.f, 0.f}));

    // Do one pass inference on this ONE piece of input
    at::Tensor y_pred = net.forward(one_input).toTensor();

    // Inspect predicted result
    for (int i = 0; i < y_pred.size(0); ++i) {
        std::cout << "Result: " << y_pred[0].item().toFloat() << std::endl;
    }

    return 0;
}