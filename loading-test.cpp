#include <torch/script.h> // One-stop header.

#include <iostream>
#include <memory>

using namespace torch::indexing;

int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: loading-test <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  std::cout << "ok\n";
  float a;
  std::cout << module.parameters().size() << std::endl;
  for (const auto& k : module.parameters()){
    std::cout << "shape: " << k.size(1) << std::endl;
    std::cout << k.index({0,0}) << std::endl;
    a = k.index({0,0}).item<float>();
  }
  std::cout << "a : " << a  << std::endl;
  //for (const auto& k : module.named_parameters()){
  //  std::cout << k[0] << std::endl;
  //}
}
