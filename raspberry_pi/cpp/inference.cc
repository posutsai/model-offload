#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <cstdint>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <opencv2/opencv.hpp>

const uint32_t batch_size = 32;

tvm::runtime::Module load_serialized_model(std::string deploy_dir) {
  tvm::runtime::Module mod_syslib = (*tvm::runtime::Registry::Get("runtime.SystemLib"))();
  std::ifstream json_in(deploy_dir + "/rpi_resnet18_graph.json");
  std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
  json_in.close();

  int device_type = kDLCPU;
  int device_id = 0;
  tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib, device_type, device_id);
  std::ifstream params_in(deploy_dir + "/rpi_resnet18_param.params", std::ios::binary);
  std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
  params_in.close();

  TVMByteArray params_arr;
  params_arr.data = params_data.c_str();
  params_arr.size = params_data.length();
  tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
  load_params(params_arr);
  return mod;
}

void forward(tvm::runtime::Module mod, cv::Mat imgs) {
  DLTensor *input;
  const int dtype_code = kDLFloat;
  const int dtype_bits = 32;
  const int dtype_lanes = 1;
  const int device_type = kDLCPU;
  const int device_id = 0;
  const int in_ndim = 4;
  const int64_t in_shape[4] = {batch_size, 3, 224, 224};
  TVMArrayAlloc(in_shape, in_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &input);//
  TVMArrayCopyFromBytes(input, imgs.data, batch_size * 224 * 3 * 224 * 4);
  tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
  tvm::runtime::PackedFunc run = mod.GetFunction("run");
  tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");
  set_input("input0", input);
  auto start = std::chrono::high_resolution_clock::now();
  run();
  auto elapsed = std::chrono::high_resolution_clock::now() - start;
  uint64_t duration = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
  printf("duration in c++ is %lu\n", duration);
  tvm::runtime::NDArray res = get_output(0);
  // 1000 for this model only
  cv::Mat output(1000, batch_size, CV_32F);
  memcpy(output.data, res->data, 1000 * batch_size * 4);
  TVMArrayFree(input);
}

void print_size(cv::Mat m, int dim) {
  printf("size is (");
  for (int i = 0; i < dim; i++) {
    printf("%3d%s", m.size[i], i != dim-1? ",": ")\n");
  }
}


int main() {
  std::ifstream img_list("./img_list.txt");
  std::string line;
  std::getline(img_list, line);
  cv::Mat batch = cv::imread(line);
  batch = cv::dnn::blobFromImage(batch, 1.0, cv::Size(224, 224), cv::Scalar(0,0,0), true);
  while (std::getline(img_list, line)) {
    if (line.empty())
      continue;
    cv::Mat img = cv::imread(line);
    img = cv::dnn::blobFromImage(img, 1.0, cv::Size(224, 224), cv::Scalar(0,0,0), true);
    batch.push_back(img);
  }
  // Relative to current working directory instead of executable
  // path.
  tvm::runtime::Module mod = load_serialized_model("../deploy");
  forward(mod, batch);
}
