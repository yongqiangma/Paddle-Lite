// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <vector>
#include "lite/api/cxx_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
#include "lite/api/test_helper.h"
#include "lite/core/op_registry.h"
#include "fstream"
#include "sstream"

namespace paddle {
namespace lite {

DEFINE_string(input_shape,
              //"1,3,224,224",
              "1,3,56,56",
              //"1,3,299,299",
              "input shapes, separated by colon and comma");

DEFINE_string(image_file,
              //"/paddle/mayongqiang01/paddle/model_test_files/img_alexnet.txt",
              "/home/nvidia/myq/model/alexnet/img_alexnet.txt",
              "read image from txt file");

//std::string model_dir = "/paddle/mayongqiang01/model/yolov3/";
std::string model_dir = "/home/nvidia/myq/model/alexnet/";
void TestModel(const std::vector<Place>& valid_places,
               const Place& preferred_place,
               bool use_npu = false) {
  Env<TARGET(kCUDA)>::Init();
  lite_api::CxxConfig cfg; 
//    auto blas = lite::cuda::Blas<float>();
  // cfg.set_model_dir(FLAGS_model_dir);
  cfg.set_model_file(model_dir+"__model__");
  cfg.set_param_file(model_dir+"__params__");
  cfg.set_preferred_place(preferred_place);
  cfg.set_valid_places(valid_places);
 /*
  std::ifstream model_f(FLAGS_model_dir + "/__model__");  
  std::stringstream buffer;
  buffer << model_f.rdbuf();
  std::string model_c(buffer.str()); 
  std::string params_c  = "hello"; 
  cfg.set_model_buffer(model_c.data(), model_c.size(), params_c.data(), params_c.size()); 
 */
  auto predictor = lite_api::CreatePaddlePredictor(cfg);
/*
  auto input_tensor = predictor->GetInput(0);
  input_tensor->Resize(std::vector<int64_t>({1, 3, 608, 608}));
  auto* data = input_tensor->mutable_data<float>();
  auto input_shape = input_tensor->shape(); 
  int item_size = std::accumulate(input_shape.begin(), input_shape.end(), 1, std::multiplies<int64_t>()); 
  for (int i = 0; i < item_size; i++) {
    data[i] = i % 255;
  }

  auto input_s = predictor->GetInput(1);
  input_s->Resize(std::vector<int64_t>({1, 2}));
  auto* s_data = input_s->mutable_data<int>();
  s_data[0] = 608;
  s_data[1] = 608;
*/
  auto split_string =
      [](const std::string& str_in) -> std::vector<std::string> {
    std::vector<std::string> str_out;
    std::string tmp_str = str_in;
    while (!tmp_str.empty()) {
      size_t next_offset = tmp_str.find(":");
      str_out.push_back(tmp_str.substr(0, next_offset));
      if (next_offset == std::string::npos) {
        break;
      } else {
        tmp_str = tmp_str.substr(next_offset + 1);
      }
    }
    return str_out;
  };

  auto get_shape = [](const std::string& str_shape) -> std::vector<int64_t> {
    std::vector<int64_t> shape;
    std::string tmp_str = str_shape;
    while (!tmp_str.empty()) {
      int dim = atoi(tmp_str.data());
      shape.push_back(dim);
      size_t next_offset = tmp_str.find(",");
      if (next_offset == std::string::npos) {
        break;
      } else {
        tmp_str = tmp_str.substr(next_offset + 1);
      }
    }
    return shape;
  };

  LOG(INFO) << "input shapes: " << FLAGS_input_shape;
  std::vector<std::string> str_input_shapes = split_string(FLAGS_input_shape);
  std::vector<std::vector<int64_t>> input_shapes;
  for (int i = 0; i < str_input_shapes.size(); ++i) {
    LOG(INFO) << "input shape: " << str_input_shapes[i];
    input_shapes.push_back(get_shape(str_input_shapes[i]));
  }
    
  for (int j = 0; j < input_shapes.size(); ++j) {
    auto input_tensor = predictor->GetInput(j);
    input_tensor->Resize(input_shapes[j]);
    auto input_data = input_tensor->mutable_data<float>();
    auto shape = input_tensor->shape();
    int item_size = 1;
    for (int i = 0; i < shape.size(); i++) {
       item_size *= shape[i];
    }
    std::ifstream read_file(FLAGS_image_file);
    for (int i = 0; i < item_size; i++) {
      read_file >> input_data[i];
//      input_data[i] = 1;
    }
   for (int i = 0; i < 10; ++i) {
     LOG(INFO)<<"input data"<< i << " "<<input_data[i];
   }
  }



  for (int i = 0; i < FLAGS_warmup; ++i) {
    predictor->Run();
  }

  auto start = GetCurrentUS();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    LOG(INFO) << i;
    predictor->Run();
  }

  LOG(INFO) << "================== Speed Report ===================";
  LOG(INFO) << "Model: " << FLAGS_model_dir << ", threads num " << FLAGS_threads
            << ", warmup: " << FLAGS_warmup << ", repeats: " << FLAGS_repeats
            << ", spend " << (GetCurrentUS() - start) / FLAGS_repeats / 1000.0
            << " ms in average.";

  auto out = predictor->GetOutput(0);

  auto out_shape = out->shape(); 
  int out_size = std::accumulate(out_shape.begin(), out_shape.end(), 1, std::multiplies<int64_t>()); 
/*
  for (int i = 0; i < out_size; i++) {
    LOG(INFO) << out->data<float>()[i];
  }
*/
  std::string  path(getcwd(NULL,0));
  std::string  locate = path +"//out.txt";
  LOG(INFO) << "out.txt:::"<<locate;
  FILE* fp = fopen(locate.c_str(), "w");
  for (int i = 0; i < out_size; i++) {
   // LOG(INFO) << "out id"<< i << " " << out[i];
    //fprintf(fp, "%f\n", (out[i]));
    fprintf(fp, "%f\n", out->data<float>()[i]);
  }
  fclose(fp);
  LOG(INFO) << "hello";
}

TEST(MobileNetV1, test_arm) {
  std::vector<Place> valid_places({
      Place{TARGET(kHost), PRECISION(kFloat)},
      Place{TARGET(kCUDA), PRECISION(kFloat)},
  });

  TestModel(valid_places, Place({TARGET(kCUDA), PRECISION(kFloat)}));
}

}  // namespace lite
}  // namespace paddle
