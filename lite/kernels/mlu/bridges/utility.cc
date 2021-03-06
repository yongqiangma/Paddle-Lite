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

#include "lite/kernels/mlu/bridges/utility.h"
#include <utility>

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

void transpose(float* input_data,
               float* output_data,
               std::vector<int> input_shape,
               std::vector<int> axis) {
  int old_index = -1;
  int new_index = -1;
  int dim[4] = {0};
  std::vector<int> shape = input_shape;
  for (dim[0] = 0; dim[0] < input_shape[0]; dim[0]++) {
    for (dim[1] = 0; dim[1] < input_shape[1]; dim[1]++) {
      for (dim[2] = 0; dim[2] < input_shape[2]; dim[2]++) {
        for (dim[3] = 0; dim[3] < input_shape[3]; dim[3]++) {
          old_index = dim[0] * shape[1] * shape[2] * shape[3] +
                      dim[1] * shape[2] * shape[3] + dim[2] * shape[3] + dim[3];
          new_index =
              dim[axis[0]] * shape[axis[1]] * shape[axis[2]] * shape[axis[3]] +
              dim[axis[1]] * shape[axis[2]] * shape[axis[3]] +
              dim[axis[2]] * shape[axis[3]] + dim[axis[3]];
          output_data[new_index] = input_data[old_index];
        }
      }
    }
  }
}

int scale2position(float scale) { return static_cast<int>(-std::log2(scale)); }

void dequant(float* dst, int8_t* src, size_t size, float scale) {
  for (size_t i = 0; i < size; ++i) {
    dst[i] = static_cast<float>(src[i]) * scale;
  }
}

void dequant(float* dst,
             int8_t* src,
             size_t size_o,
             size_t size,
             size_t size_in,
             std::vector<float> scales) {
  for (int out = 0; out < size_o; ++out) {
    for (int s = 0; s < size; ++s) {
      auto scale = scales[s];
      for (int in = 0; in < size_in; ++in) {
        int idx = in + s * size_in + out * size_in * size;
        dst[idx] = static_cast<float>(src[idx]) * scale;
      }
    }
  }
}

cnmlActiveFunction_t OpTypeToCNMLActType(std::string op_type) {
  if (op_type == "relu") {
    return CNML_ACTIVE_RELU;
  } else if (op_type == "sigmoid") {
    return CNML_ACTIVE_SIGMOID;
  } else if (op_type == "tanh") {
    return CNML_ACTIVE_TANH;
  } else if (op_type == "relu1") {
    return CNML_ACTIVE_RELU1;
  } else if (op_type == "relu6") {
    return CNML_ACTIVE_RELU6;
  } else if (op_type == "hard_sigmoid") {
    return CNML_ACTIVE_HARD_SIGMOID;
  }
  LOG(FATAL) << "CNML Unspoorted op type " << op_type;
  return CNML_ACTIVE_NONE;
}

bool HasInputArg(const OpInfo* op_info,
                 const Scope* scope,
                 const std::string& argname) {
  auto iarg_names = op_info->input_argnames();
  if (std::find(iarg_names.begin(), iarg_names.end(), argname) !=
      iarg_names.end()) {
    auto inputs = op_info->Input(argname);
    if (inputs.empty()) {
      return false;
    }
    auto var_name = inputs.front();
    auto var = scope->FindVar(var_name);
    return var != nullptr;
  } else {
    return false;
  }
}
}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
