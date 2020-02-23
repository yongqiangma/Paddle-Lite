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

#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lite/api/test_helper.h"
#include "lite/backends/vulkan/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/vulkan/vulkan_utils.h"
#include "lite/tests/utils/naive_math_impl.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace vulkan {

void fill_bias_fc(float* out, const float* bias, int num, int channel) {
  int remain = channel;
  for (int j = 0; j < num; ++j) {
    const float* ptr_bias = bias;
    float* ptr_out = out + j * channel;
    for (int i = 0; i < remain; ++i) {
      *(ptr_out++) += *(ptr_bias++);
    }
  }
}

DDim compute_out_dim(const DDim& dim_in, const DDim& wdim, int in_num_col_dim) {
  std::vector<int64_t> out_dim;
  out_dim.resize(in_num_col_dim + 1);
  auto in_mat_dims = dim_in.Flatten2D(in_num_col_dim);
  for (int i = 0; i < in_num_col_dim; ++i) {
    out_dim[i] = dim_in[i];
  }
  out_dim[in_num_col_dim] = wdim[1];
  return DDim(out_dim);
}

void test_vulkan_fc_fp32_16(DDim in_dims, DDim w_dims, DDim b_dims) {
  LOG(INFO) << "vulkan_fc test";

  auto kernels = KernelRegistry::Global().Create(
      "fc", TARGET(kVULKAN), PRECISION(kFloat), DATALAYOUT(kNCHW));

  ASSERT_FALSE(kernels.empty());

  auto kernel = std::move(kernels.front());

  auto context = ContextScheduler::Global().NewContext(TARGET(kVULKAN));

  lite::Tensor input, input_vk, output, weight, weight_vk, bias, bias_vk,
      input_im, out_buf, out_buf_vk, output_ref;

  auto device = context->As<VULKANContext>().device();

  operators::FcParam param;
  // fill param

  int in_num_col_dims_ = 1;
  param.input = &input_im;
  param.w = &weight_vk;
  param.bias = &bias_vk;
  param.output = &output;
  param.in_num_col_dims = 1;

  bool flag_bias = b_dims.production() < 1 ? false : true;

  kernel->SetParam(param);

  std::unique_ptr<KernelContext> dep_context(new KernelContext);
  context->As<VULKANContext>().CopySharedTo(
      &(dep_context->As<VULKANContext>()));
  kernel->SetContext(std::move(dep_context));

  input.Resize(in_dims);
  input_im.Resize(in_dims);
  weight.Resize(w_dims);
  bias.Resize(b_dims);

  auto* input_v = input.mutable_data<float>();

  for (int i = 0; i < in_dims.production(); i++) {
    input_v[i] = i % 3;
    // LOG(INFO)<<"in:"<<input_v[i];
  }

  auto* weight_v = weight.mutable_data<float>();

  for (int i = 0; i < w_dims.production(); i++) {
    weight_v[i] = i % 5;
  }

  auto* bias_v = bias.mutable_data<float>();

  for (int i = 0; i < b_dims.production(); i++) {
    bias_v[i] = i % 7;
    // LOG(INFO)<<"b_data: "<<i<<"  :"<<bias_v[i];
  }
  DDim out_dim = compute_out_dim(in_dims, w_dims, in_num_col_dims_);
  output.Resize(out_dim);

  CopyToVulkan(device, &input, &input_vk);
  VulkanBuf2Image(device, &input_vk, &input_im, BufTYPE::NCHW);

  VulkanPrintImage(device, &input_im, BufTYPE::NCHW);

  CopyToVulkan(device, &weight, &weight_vk);
  CopyToVulkan(device, &bias, &bias_vk);

  kernel->Launch();

  int FLAGS_repeats = 5;
  auto start = GetCurrentUS();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    VulkanRun(device);
  }
  auto time = (GetCurrentUS() - start) / FLAGS_repeats / 1000.0;
  LOG(INFO) << "H:" << in_dims << "time:" << time;

  CopyToHost(device, &output, &out_buf);
  auto* out_cpu = out_buf.data<float>();

  output_ref.Resize(out_dim);
  auto output_ref_data = output_ref.mutable_data<float>();

  int m = input.dims().count(0, in_num_col_dims_);
  CHECK_EQ(w_dims[0],
           input.dims().count(in_num_col_dims_, input.dims().size()));
  int k = w_dims[0];
  int n = w_dims[1];
  if (m == 1) {
    basic_gemv(n,
               k,
               weight_v,
               input_v,
               bias_v,
               output_ref_data,
               1.f,
               0.f,
               true,
               flag_bias,
               false);
  } else {
    basic_gemm(false,
               false,
               m,
               n,
               k,
               1.f,
               input_v,
               k,
               weight_v,
               n,
               0.f,
               output_ref_data,
               n,
               bias_v,
               false,
               false);
    if (flag_bias) {
      fill_bias_fc(output_ref_data, bias_v, m, n);
    }
  }
  // compare
  /*
  for (int i = 0; i < output.dims().production(); i++) {
    LOG(INFO) << i << " out:" << out_cpu[i] << "ref:" << output_ref_data[i];
  }
  */
  float sum = 0;
  for (int i = 0; i < output.dims().production(); i++) {
    float gap = output_ref_data[i] * 2 / 1000;
    gap = gap < 1e-4 ? 1e-4 : gap;
    sum += output_ref_data[i];
    EXPECT_NEAR(out_cpu[i], output_ref_data[i], gap);
  }
}
TEST(fc_vulkan, compute) {
  if (false) {
    for (auto& m : {1}) {
      for (auto& n : {1, 4, 16, 128, 256, 1024}) {
        for (auto& k : {1, 16, 128, 1024}) {
          for (auto& bflag : {true}) {
            DDim dim_in{{m, k, 1, 1}};
            DDim wdim{{k, n}};
            DDim bdim{{bflag ? n : 0}};

            LOG(INFO) << "run m: " << m << ", n: " << n << ", k: " << k
                      << ", bias: " << (bflag ? "true" : "false") << "failed";
            test_vulkan_fc_fp32_16(dim_in, wdim, bdim);
          }
        }
      }
    }
  }

  LOG(INFO) << "run m: ";
  if (true) {
    for (auto& m : {1}) {
      for (auto& n : {4}) {
        for (auto& k : {4}) {
          // DDim dim_in{{m, k,1,1}};
          DDim dim_in{{m, k, 1, 1}};
          DDim wdim{{k, n}};
          bool bflag = true;  // fc bias is always true
          DDim bdim{{bflag ? n : 0}};
          // DDim bdim{{1, n }};

          DDim temp{{1, 2, 3, 4}};
          LOG(INFO) << temp;
          LOG(INFO) << temp[0] << "  " << temp[1];
          LOG(INFO) << "run m: " << m << ", n: " << n << ", k: " << k
                    << ", bias: " << (bflag ? "true" : "false") << " failed";
          test_vulkan_fc_fp32_16(dim_in, wdim, bdim);
        }
      }
    }
  }
}

}  // namespace vulkan
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
USE_LITE_KERNEL(fc, kVULKAN, kFloat, kNCHW, def);
