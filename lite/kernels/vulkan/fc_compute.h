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

#pragma once
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "lite/backends/vulkan/vk_buffer.h"
#include "lite/backends/vulkan/vk_device.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/vulkan/vulkan_utils.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace vulkan {

template <typename Dtype>
void naive_transpose(const Dtype* din, Dtype* dout, int m, int n) {
  int k = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      dout[k++] = din[j * n + i];
    }
  }
}

template <PrecisionType PType>
void fc_trans_weights(const Tensor& tin, Tensor* tout);

template <>
void fc_trans_weights<PRECISION(kFloat)>(const Tensor& tin, Tensor* tout) {
  CHECK_EQ(tin.dims().size(), 2) << "fc weights size must = 2";
  int m = tin.dims()[0];
  int n = tin.dims()[1];
  tout->Resize({n, m});
  auto ptr_in = tin.data<float>();
  auto ptr_out = tout->mutable_data<float>();
  naive_transpose(ptr_in, ptr_out, m, n);
}

bool check_fc_use_gemm(int m, const std::vector<float>& scale, bool has_bias) {
  return m > 1;
}

class FcCompute : public KernelLite<TARGET(kVULKAN), PRECISION(kFloat)> {
 public:
  using param_t = operators::FcParam;

  virtual void ReInitWhenNeeded() {
    // param = this->template Param<operators::FcParam>();
    auto& param = this->template Param<operators::FcParam>();
    auto x_dims = param.input->dims();
    if (last_shape_ == x_dims) {
      return;
    }
    last_shape_ = x_dims;
    auto w_dims = param.w->dims();
    auto& ctx = this->ctx_->template As<VULKANContext>();
    CHECK_GE(x_dims.size(), 2UL);
    CHECK_EQ(w_dims.size(), 2UL);
    CHECK_EQ(param.output->dims().size(), 2UL);

    m_ = x_dims.Slice(0, param.in_num_col_dims).production();
    k_ = x_dims.Slice(param.in_num_col_dims, x_dims.size()).production();
    CHECK_EQ(k_, w_dims[0]);
    n_ = w_dims[1];
    CHECK_EQ(k_, static_cast<int>(w_dims[0]));
    flag_gemm_ =
        check_fc_use_gemm(m_, param.weight_scale, param.bias != nullptr);

    // if (!flag_trans_weights_ && !flag_gemm_) {
    // if (!flag_trans_weights_ && !flag_gemm_) {
    //   flag_trans_weights_ = true;
    //     fc_trans_weights(*param.w, &weights_);
    //}
  }

  void PrepareForRun() override;

  void Run() override;

  virtual ~FcCompute() = default;

 private:
  DDim last_shape_;
  Tensor weights_;
  Tensor bias_;
  bool flag_trans_weights_{false};
  bool flag_trans_bias_{false};
  bool flag_gemm_{true};
  int m_;
  int n_;
  int k_;
  operators::FcParam param;
  std::vector<float> scale_;
  std::vector<VkCommandBuffer> cmds;
  std::shared_ptr<lite::vulkan::VulkanDevice> device;
};

}  // namespace vulkan
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
