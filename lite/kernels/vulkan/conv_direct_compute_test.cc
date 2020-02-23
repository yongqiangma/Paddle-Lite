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

void test_vulkan_conv_fp32_16(const std::vector<DDim>& input_dims,
                              const DDim& weights_dim,
                              int group,
                              const std::vector<int>& strides,
                              const std::vector<int>& pads,
                              const std::vector<int>& dilas,
                              bool flag_bias,
                              bool flag_relu,
                              std::string conv_name) {
  LOG(INFO) << "vulkan_conv test";

  // auto kernels = KernelRegistry::Global().Create(
  //     "conv_direct", TARGET(kVULKAN), PRECISION(kFloat), DATALAYOUT(kNCHW));

  auto kernels = KernelRegistry::Global().Create(
      conv_name, TARGET(kVULKAN), PRECISION(kFloat), DATALAYOUT(kNCHW));

  auto kernels_io = KernelRegistry::Global().Create(
      "io_copy", TARGET(kVULKAN), PRECISION(kFloat), DATALAYOUT(kNCHW));

  ASSERT_FALSE(kernels.empty());

  auto kernel = std::move(kernels.front());

  auto context = ContextScheduler::Global().NewContext(TARGET(kVULKAN));

  lite::Tensor input, output, filter, filter_vk, input_im, bias, bias_vk,
      input_vk, out_buf_vk;

  auto device = context->As<VULKANContext>().device();

  operators::ConvParam param_;
  param_.x = &input_im;
  param_.filter = &filter_vk;
  param_.output = &output;
  param_.paddings = pads;
  param_.strides = strides;
  param_.dilations = dilas;
  param_.bias = &bias_vk;
  param_.fuse_relu = flag_relu;
  kernel->SetParam(param_);

  std::unique_ptr<KernelContext> dep_context(new KernelContext);
  context->As<VULKANContext>().CopySharedTo(
      &(dep_context->As<VULKANContext>()));
  kernel->SetContext(std::move(dep_context));

  lite::DDim input_dim = input_dims[0];
  input.Resize(input_dims[0]);
  auto* input_v = input.mutable_data<float>();
  auto data_in = input.data<float>();

  for (int i = 0; i < input_dim.production(); i++) {
    // input_v[i] = i/10;
    // if(input_v[i] > 30){
    input_v[i] = 1;
  }

  filter.Resize(weights_dim);
  auto* filter_data = filter.mutable_data<float>();
  for (int i = 0; i < weights_dim.production(); i++) {
    // filter_data[i] = rand()%100/(float)87.0 ;
    filter_data[i] = i % 7;
    // LOG(INFO)<<filter_data[i];
  }

  int dila_h = dilas[0];
  int dila_w = dilas[1];
  int kernel_h = weights_dim[2];
  auto kernel_exten = dila_h * (kernel_h - 1) + 1;
  int stride_h = strides[0];
  int stride_w = strides[1];
  int h_out = (input_dim[2] + 2 * pads[0] - kernel_exten) / stride_h + 1;
  int w_out = (input_dim[3] + 2 * pads[1] - kernel_exten) / stride_w + 1;
  lite::DDim output_dim = lite::DDim{
      std::vector<int64_t>({input_dim[0], weights_dim[0], h_out, w_out})};
  if (h_out < 1 || w_out < 1) {
    return;
  }
  output.Resize(output_dim);
  bias.Resize({weights_dim[0]});
  bias_vk.Resize({weights_dim[0]});
  auto* bias_data = bias.mutable_data<float>();
  memset(reinterpret_cast<void*> bias_data, 0, weights_dim[0] * sizeof(float));
  if (flag_bias) {
    for (int i = 0; i < weights_dim[0]; i++) {
      bias_data[i] = i % 10;
    }
  }

  CopyToVulkan(device, &input, &input_vk);

  VulkanBuf2Image(device, &input_vk, &input_im, BufTYPE::NCHW);

  // operators::IoCopyParam ioparam_;
  // ioparam_.x = &filter;
  // ioparam_.y = &filter_vk;
  // kernels_io->Launch();
  CopyToVulkan(device, &filter, &filter_vk);

  CopyToVulkan(device, &bias, &bias_vk);

  kernel->Launch();
  lite::Tensor out_buf;

  int FLAGS_repeats = 5;
  auto start = GetCurrentUS();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    VulkanRun(device);
  }
  auto time = (GetCurrentUS() - start) / FLAGS_repeats / 1000.0;
  LOG(INFO) << "H:" << input_dims[0] << "time:" << time;

  VulkanImage2Buf(device, &output, &out_buf_vk, BufTYPE::NCHW);
  CopyToHost(device, &out_buf_vk, &out_buf);
  auto* out_cpu = out_buf.mutable_data<float>();

  std::vector<float> out_ref(out_buf.numel());

  auto dim_in = input.dims().data();
  auto dim_out = out_buf.dims().data();
  auto wptr = filter_data;
  auto bias_ptr = bias.data<float>();
  for (int i = 0; i < weights_dim[0]; i++) {
    // LOG(INFO) << "bias_ptr :"<<bias_ptr[i];
  }
  auto weight_dim = filter.dims().data();
  conv_basic<float, float>(input.data<float>(),
                           out_ref.data(),
                           dim_in[0],
                           dim_out[1],
                           dim_out[2],
                           dim_out[3],
                           dim_in[1],
                           dim_in[2],
                           dim_in[3],
                           wptr,
                           bias_ptr,
                           group,
                           weight_dim[3],
                           weight_dim[2],
                           strides[0],
                           strides[1],
                           dilas[0],
                           dilas[1],
                           pads[0],
                           pads[1],
                           flag_bias,
                           flag_relu);
  // output data
  /*
   for (int i = 0; i < out_buf.numel(); i++) {
     LOG(INFO) << i << "  out_cpu: " << (reinterpret_cast<float*>(out_cpu))[i]
               << " out_ref: " << out_ref[i];
   }

   for (int i = 0; i < out_buf.numel(); i++) {
     EXPECT_NEAR((reinterpret_cast<float*>(out_cpu))[i],
                 out_ref[i],
                 abs(2 * out_ref[i] / 1000));
   }
 */
  LOG(INFO) << "SUCCESS!";
}

TEST(TestConvDirect, test_conv_direct) {
  if (false) {
    for (auto& cin : {256}) {
      // for (auto& cin : {5}) {
      for (auto& cout : {256}) {
        // for (auto& cout : {5}) {
        for (auto& pad : {0}) {
          for (auto& flag_bias : {true}) {
            for (auto& flag_relu : {true}) {
              std::vector<DDim> dims;
              DDim weights_dim({cout, cin, 1, 1});
              // DDim weights_dim({cout, cin, 2, 2});
              for (auto& batch : {1}) {
                for (auto& h : {112, 56, 28, 14}) {
                  dims.push_back(DDim({batch, cin, h, h}));
                  LOG(INFO) << "dims:" << dims[0] << "  weights_dim"
                            << weights_dim;
                  test_vulkan_conv_fp32_16(dims,
                                           weights_dim,
                                           1,
                                           {1, 1},
                                           {pad, pad},
                                           {1, 1},
                                           flag_bias,
                                           flag_relu,
                                           "conv2d");
                  dims.clear();
                }
              }
            }
          }
        }
      }
    }
  }
  if (false) {
    std::vector<DDim> dims;
    int h, w;
    h = w = 112;
    int c = 3;
    dims.push_back(DDim({1, c, h, w}));
    DDim weights_dim({1, 3, 3, 3});

    int stride = 2;
    int pad = 1;
    bool flag_bias = true;
    bool flag_relu = true;
    test_vulkan_conv_fp32_16(dims,
                             weights_dim,
                             1,
                             {stride, stride},
                             {pad, pad},
                             {1, 1},
                             flag_bias,
                             flag_relu,
                             "conv2d");
  }
  if (false) {
    std::vector<DDim> dims;
    int h, w;
    h = w = 112;
    int c = 32;
    dims.push_back(DDim({1, c, h, w}));
    DDim weights_dim({32, 1, 3, 3});

    int stride = 2;
    int pad = 1;
    bool flag_bias = true;
    bool flag_relu = true;
    test_vulkan_conv_fp32_16(dims,
                             weights_dim,
                             c,
                             {stride, stride},
                             {pad, pad},
                             {1, 1},
                             flag_bias,
                             flag_relu,
                             "depthwise_conv2d");
    LOG(INFO) << "dims:" << dims[0] << "  weights_dim" << weights_dim
              << " stride:" << stride << " pad:" << pad << " bias:" << flag_bias
              << " relu" << flag_relu;
  }
}
TEST(TestConv_dw, test_conv_dw) {
  if (false) {
    for (auto& stride : {1}) {
      for (auto& pad : {1}) {
        for (auto& flag_bias : {true}) {
          for (auto& flag_relu : {true}) {
            for (auto& c : {128}) {
              std::vector<DDim> dims;
              DDim weights_dim({c, 1, 3, 3});
              for (auto& batch : {1}) {
                for (auto& h : {112, 56, 28, 14}) {
                  // for (auto& h : { 14}) {
                  dims.push_back(DDim({batch, c, h, h}));
                  LOG(INFO) << "-------------------------------------";
                  LOG(INFO) << "dims:" << dims[0] << "  weights_dim"
                            << weights_dim << " stride:" << stride
                            << " pad:" << pad;
                  test_vulkan_conv_fp32_16(dims,
                                           weights_dim,
                                           c,
                                           {stride, stride},
                                           {pad, pad},
                                           {1, 1},
                                           flag_bias,
                                           flag_relu,
                                           "depthwise_conv2d");
                  dims.clear();
                }
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace vulkan
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
USE_LITE_KERNEL(depthwise_conv2d, kVULKAN, kFloat, kNCHW, def);
USE_LITE_KERNEL(io_copy, kVULKAN, kFloat, kNCHW, device_to_host);
USE_LITE_KERNEL(io_copy, kVULKAN, kFloat, kNCHW, host_to_device);
USE_LITE_KERNEL(conv2d, kVULKAN, kFloat, kNCHW, def);
USE_LITE_KERNEL(buf2image, kVULKAN, kFloat, kNCHW, def);
USE_LITE_KERNEL(image2buf, kVULKAN, kFloat, kNCHW, def);
