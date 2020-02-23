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
#include "lite/backends/vulkan/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/vulkan/activation_compute.h"
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
                              bool flag_relu) {
  LOG(INFO) << "vulkan_conv test";

  auto kernels_input = KernelRegistry::Global().Create(
      "buf2image", TARGET(kVULKAN), PRECISION(kFloat), DATALAYOUT(kNCHW));

  auto kernels = KernelRegistry::Global().Create(
      "conv", TARGET(kVULKAN), PRECISION(kFloat), DATALAYOUT(kNCHW));
  auto kernels_output = KernelRegistry::Global().Create(
      "image2buf", TARGET(kVULKAN), PRECISION(kFloat), DATALAYOUT(kNCHW));
  ASSERT_FALSE(kernels_input.empty());
  ASSERT_FALSE(kernels.empty());
  ASSERT_FALSE(kernels_output.empty());

  auto kernel_input = std::move(kernels_input.front());
  auto kernel = std::move(kernels.front());
  auto kernel_output = std::move(kernels_output.front());

  lite::Tensor input, input2image, output, output2buf, filter, input_im;

  operators::ActivationParam param_input;
  param_input.X = &input;
  param_input.Out = &input2image;

  operators::ConvParam param_;
  param_.x = &input_im;
  param_.filter = &filter;
  param_.output = &output;

  operators::ActivationParam param_output;
  param_output.X = &output;
  param_output.Out = &output2buf;

  auto context = ContextScheduler::Global().NewContext(TARGET(kVULKAN));
  auto device = context->As<VULKANContext>().device();

  kernel_input->SetParam(param_input);

  kernel->SetParam(param_);
  kernel_output->SetParam(param_output);

  std::unique_ptr<KernelContext> dep_context_input(new KernelContext);
  context->As<VULKANContext>().CopySharedTo(
      &(dep_context_input->As<VULKANContext>()));
  kernel_input->SetContext(std::move(dep_context_input));

  std::unique_ptr<KernelContext> dep_context(new KernelContext);
  context->As<VULKANContext>().CopySharedTo(
      &(dep_context->As<VULKANContext>()));
  kernel->SetContext(std::move(dep_context));

  std::unique_ptr<KernelContext> dep_context_output(new KernelContext);
  context->As<VULKANContext>().CopySharedTo(
      &(dep_context_output->As<VULKANContext>()));
  kernel_output->SetContext(std::move(dep_context_output));

  // lite::DDim input_dim = input_dims[0];
  // std::vector<float> input_v(input_dim.production());
  // for (int i = 0; i < input_v.size(); i++) {
  //   input_v[i] = i;
  // }

  lite::DDim kernel_dim = weights_dim;
  std::vector<float> kernel_v(kernel_dim.production());
  for (int i = 0; i < kernel_v.size(); i++) {
    kernel_v[i] = (kernel_v.size() - i) / 10;
  }

  // input.Assign<float, lite::DDim, TARGET(kVULKAN)>(input_v.data(),
  // input_dim);

  lite::DDim input_dim = input_dims[0];
  input.Resize(input_dims[0]);
  auto* input_v = input.mutable_data<float>();
  auto data_in = input.data<float>();

  for (int i = 0; i < input_dim.production(); i++) {
    // input_v[i] = i/10;
    // if(input_v[i] > 30){
    input_v[i] = i % 30;
    // }
    LOG(INFO) << input_v[i];
    // LOG(INFO)<<"data_in"<<data_in[i];
  }

  VulkanBuf2Image(device, &input, &input_im, BufTYPE::HCHW);
  LOG(INFO) << "ccccc  cccccc";
  // filter.Assign<float, lite::DDim, TARGET(kVULKAN)>(kernel_v.data(),
  //                                                   kernel_dim);
  filter.Resize(weights_dim);
  auto* filter_data = filter.mutable_data<float>();
  for (int i = 0; i < weights_dim.production(); i++) {
    filter_data[i] = (weights_dim.production() - i) / 10;
    // if(filter_data[i] > 30){
    filter_data[i] = i % 30;
    // }
  }
  input2image.Resize(input_dim);
  // input_im.Resize(input_dim);
  int dila_h = dilas[0];
  int dila_w = dilas[1];
  int kernel_h = kernel_dim[2];
  auto kernel_exten = dila_h * (kernel_h - 1) + 1;
  int stride_h = strides[0];
  int stride_w = strides[1];
  int h_out = (input_dim[2] + 2 * pads[0] - kernel_exten) / stride_h + 1;
  int w_out = (input_dim[3] + 2 * pads[1] - kernel_exten) / stride_w + 1;
  lite::DDim output_dim = lite::DDim{
      std::vector<int64_t>({input_dim[0], kernel_dim[0], h_out, w_out})};
  LOG(INFO) << "output_dim" << input_dim[0] << "   " << kernel_dim[0] << "   "
            << h_out << "   " << w_out;

  output.Resize(output_dim);
  output2buf.Resize(output_dim);

  std::vector<float> bias_v(output_dim.production());
  for (int i = 0; i < bias_v.size(); i++) {
    bias_v[i] = i / 10;
  }

  // kernel_input->Launch();

  kernel->Launch();
  LOG(INFO) << "ccccc  cccccc kernel";

  kernel_output->Launch();

  lite::vulkan::VulkanBuf* out_buf2 =
      (lite::vulkan::VulkanBuf*)output2buf.data<float>();
  float* out_cpu_data2 =
      static_cast<float*>(malloc(sizeof(float) * output2buf.numel()));
  CopySync<TARGET(kVULKAN)>(reinterpret_cast<void*> out_cpu_data2,
                            reinterpret_cast<void*> out_buf2,
                            sizeof(float) * output2buf.numel(),
                            IoDirection::DtoH);
  for (int i = 0; i < output2buf.numel(); i++) {
    LOG(INFO) << "index:" << i << ": "
              << (reinterpret_cast<float*>(out_cpu_data2))[i];
  }

  std::vector<float> out_ref(output2buf.numel());
  // for (int i = 0; i < input_v.size(); i++) {
  //   out_ref[i] = std::max(0.f, input_v[i]);
  // }
  auto dim_in = input.dims().data();
  auto dim_out = output2buf.dims().data();
  auto wptr = kernel_v.data();
  auto bias_ptr = bias_v.data();
  int group = dim_in[1];
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

  for (int i = 0; i < output2buf.numel(); i++) {
    LOG(INFO) << "out_ref"
              << " i: " << out_ref[i];
  }
  for (int i = 0; i < output2buf.numel(); i++) {
    EXPECT_NEAR((reinterpret_cast<float*>(out_cpu_data2))[i],
                out_ref[i],
                out_ref[i] / 1000);
  }
  LOG(INFO) << "SUCCESS!";
}

TEST(TestVulkanConvDW, test_conv_DW) {
  // if (true) {
  //   for (auto& stride : {1}) {
  //     for (auto& pad : {0}) {
  //       for (auto& flag_bias : {false}) {
  //         for (auto& flag_relu : {false}) {
  //           for (auto& c : { 3}) {
  //             std::vector<DDim> dims;
  //             DDim weights_dim({c, 1, 3, 3});
  //             for (auto& batch : {1}) {
  //               for (auto& h : {6}) {
  //                 dims.push_back(DDim({batch, c, h, h}));
  //               }
  //             }
  //             test_vulkan_conv_fp32_16(dims,
  //                            weights_dim,
  //                            c,
  //                            {stride, stride},
  //                            {pad, pad},
  //                            {1, 1},
  //                            flag_bias,
  //                            flag_relu);
  //           }
  //         }
  //       }
  //     }
  //   }
  // }
  std::vector<DDim> dims;
  int h, w;
  h = w = 6;
  int c = 3;
  dims.push_back(DDim({1, c, h, w}));
  DDim weights_dim({c, 1, 3, 3});
  int stride = 1;
  int pad = 0;
  bool flag_bias = false;
  bool flag_relu = false;
  test_vulkan_conv_fp32_16(dims,
                           weights_dim,
                           c,
                           {stride, stride},
                           {pad, pad},
                           {1, 1},
                           flag_bias,
                           flag_relu);
}

}  // namespace vulkan
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
USE_LITE_KERNEL(conv, kVULKAN, kFloat, kNCHW, def);
USE_LITE_KERNEL(buf2image, kVULKAN, kFloat, kNCHW, def);
USE_LITE_KERNEL(image2buf, kVULKAN, kFloat, kNCHW, def);
