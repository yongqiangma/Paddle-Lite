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
#include "lite/kernels/vulkan/vulkan_utils.h"
#include "lite/tests/utils/naive_math_impl.h"
namespace paddle {
namespace lite {
namespace kernels {
namespace vulkan {

int PoolOutputSize(
    int input_size, int filter_size, int padding, int stride, bool ceil_mode) {
  int output_size;
  if (!ceil_mode) {
    output_size = (input_size - filter_size + 2 * padding) / stride + 1;
  } else {
    output_size =
        (input_size - filter_size + 2 * padding + stride - 1) / stride + 1;
  }
  return output_size;
}

std::vector<int64_t> compute_output_shape(operators::PoolParam* param_) {
  const auto x_dims = param_->x->dims();
  std::vector<int>& ksize = param_->ksize;
  if (param_->global_pooling) {
    ksize.resize(static_cast<size_t>(x_dims.size()) - 2);
    for (size_t i = 0; i < ksize.size(); ++i) {
      param_->paddings[i] = 0;
      ksize[i] = static_cast<int>(x_dims[i + 2]);
    }
  }

  std::vector<int64_t> output_shape({x_dims[0], x_dims[1]});
  if (param_->adaptive) {
    output_shape.insert(
        output_shape.end(), param_->ksize.begin(), param_->ksize.end());
  } else {
    for (size_t i = 0; i < param_->ksize.size(); ++i) {
      output_shape.push_back(PoolOutputSize(x_dims[i + 2],
                                            param_->ksize[i],
                                            param_->paddings[i],
                                            param_->strides[i],
                                            param_->ceil_mode));
    }
  }
  return output_shape;
}

void pool_compute_ref(const operators::PoolParam& param) {
  auto& in_dims = param.x->dims();
  auto& out_dims = param.output->dims();

  const float* src_ptr = param.x->data<const float>();
  float* dst_ptr = param.output->mutable_data<float>();

  std::vector<int> ksize = param.ksize;
  std::vector<int> strides = param.strides;
  std::vector<int> paddings = param.paddings;

  std::string pooling_type = param.pooling_type;
  bool global_pooling = param.global_pooling;
  bool exclusive = param.exclusive;
  bool adaptive = param.adaptive;
  bool ceil_mode = param.ceil_mode;
  bool use_quantizer = param.use_quantizer;
  std::string data_format = param.data_format;

  int in_n = in_dims[0];
  int in_c = in_dims[1];
  int in_h = in_dims[2];
  int in_w = in_dims[3];
  int size_in_n = in_c * in_h * in_w;
  int size_in_c = in_h * in_w;

  int out_h = out_dims[2];
  int out_w = out_dims[3];
  int size_out_n = in_c * out_h * out_w;
  int size_out_c = out_h * out_w;

  int window_h = ksize[0];
  int window_w = ksize[1];
  int stride_h = strides[0];
  int stride_w = strides[1];
  int pad_h = paddings[0];
  int pad_w = paddings[1];

  if (global_pooling == true) {
    for (int n = 0; n < in_n; ++n) {
      for (int c = 0; c < in_c; ++c) {
        const float* src = src_ptr + n * size_in_n + c * size_in_c;
        float res = src[0];
        if (pooling_type == "max") {
          for (int i = 1; i < size_in_c; ++i) {
            float cur_val = src[i];
            res = cur_val > res ? cur_val : res;
          }
        } else if (pooling_type == "avg") {
          for (int i = 1; i < size_in_c; ++i) {
            float cur_val = src[i];
            res += cur_val;
          }
          res /= size_in_c;
        }
        dst_ptr[n * size_out_n + c] = res;
      }
    }
  } else {
    for (int n = 0; n < in_n; ++n) {
      for (int c = 0; c < in_c; ++c) {
        for (int h = 0; h < out_h; ++h) {
          int sh = h * stride_h;
          int eh = sh + window_h;
          sh = (sh - pad_h) < 0 ? 0 : sh - pad_h;
          eh = (eh - pad_h) > in_h ? in_h : eh - pad_h;
          for (int w = 0; w < out_w; ++w) {
            int sw = w * stride_w;
            int ew = sw + window_w;
            sw = (sw - pad_w) < 0 ? 0 : sw - pad_w;
            ew = (ew - pad_w) > in_w ? in_w : ew - pad_w;
            int pooling_size = (ew - sw) * (eh - sh);
            if (pooling_size == 0) continue;
            float res = 0.f;
            for (int kh = sh; kh < eh; ++kh) {
              for (int kw = sw; kw < ew; ++kw) {
                int src_idx = n * size_in_n + c * size_in_c + kh * in_w + kw;
                if (kh == sh && kw == sw) {
                  res = src_ptr[src_idx];
                } else {
                  if (pooling_type == "max") {
                    res = res >= src_ptr[src_idx] ? res : src_ptr[src_idx];
                  }
                  if (pooling_type == "avg") {
                    res += src_ptr[src_idx];
                  }
                }
              }
            }
            if (pooling_type == "avg") {
              if (exclusive) {
                res /= pooling_size;
              } else {
                res /= window_h * window_w;
              }
            }
            dst_ptr[n * size_out_n + c * size_out_c + h * out_w + w] = res;
          }
        }
      }
    }
  }
}

void test_vulkan_pool2d_fp32_16(const std::vector<DDim>& input_dims,
                                std::string pooling_type,
                                const bool global_pooling,
                                int ksize,
                                const std::vector<int>& strides,
                                const std::vector<int>& pads,
                                bool exclusive,
                                const bool ceil_mode) {
  LOG(INFO) << "vulkan_pool test";

  auto kernels = KernelRegistry::Global().Create(
      "pool2d", TARGET(kVULKAN), PRECISION(kFloat), DATALAYOUT(kNCHW));

  ASSERT_FALSE(kernels.empty());

  auto kernel = std::move(kernels.front());

  auto context = ContextScheduler::Global().NewContext(TARGET(kVULKAN));

  lite::Tensor input, output, input_im, input_vk, out_buf, output_ref,
      oub_buf_vk;

  auto device = context->As<VULKANContext>().device();

  operators::PoolParam param;
  // fill param
  param.x = &input_im;
  param.output = &output;
  param.pooling_type = pooling_type;

  param.ksize = {ksize, ksize};

  param.global_pooling = global_pooling;
  param.strides = strides;
  param.paddings = pads;
  param.exclusive = exclusive;
  param.ceil_mode = ceil_mode;
  param.adaptive = false;
  param.use_quantizer = false;
  kernel->SetParam(param);

  std::unique_ptr<KernelContext> dep_context(new KernelContext);
  context->As<VULKANContext>().CopySharedTo(
      &(dep_context->As<VULKANContext>()));
  kernel->SetContext(std::move(dep_context));

  lite::DDim input_dim = input_dims[0];
  input.Resize(input_dims[0]);
  input_im.Resize(input_dims[0]);

  auto* input_v = input.mutable_data<float>();

  for (int i = 0; i < input_dim.production(); i++) {
    input_v[i] = i;
  }

  std::vector<int64_t> output_dim = compute_output_shape(&param);
  output.Resize(output_dim);

  CopyToVulkan(device, &input, &input_vk);

  VulkanBuf2Image(device, &input_vk, &input_im, BufTYPE::NCHW);
  // VulkanPrintImage(device,&input_im);

  kernel->Launch();
  // VulkanPrintImage(device,&output);
  VulkanRun(device);
  VulkanImage2Buf(device, &output, &out_buf_vk, BufTYPE::NCHW);

  CopyToHost(device, &out_buf_vk, &out_buf);
  auto* out_cpu = out_buf.mutable_data<float>();

  std::vector<float> out_ref(out_buf.numel());

  auto dim_in = input.dims().data();
  auto dim_out = out_buf.dims().data();

  output_ref.Resize(DDim(output_dim));

  auto* output_ref_data = output_ref.mutable_data<float>();
  for (int i = 0; i < output.dims().production(); ++i) {
    output_ref_data[i] = -2;
  }

  operators::PoolParam refparam(param);
  refparam.x = &input;
  refparam.output = &output_ref;

  pool_compute_ref(refparam);

  for (int i = 0; i < out_buf.numel(); i++) {
    LOG(INFO) << "dim_outm, " << i << ": vulkan: " << out_cpu[i]
              << " ref:" << output_ref_data[i];
  }

  // compare
  for (int i = 0; i < output.dims().production(); i++) {
    EXPECT_NEAR(out_cpu[i], output_ref_data[i], output_ref_data[i] * 2 / 1000);
  }
  LOG(INFO) << "SUCCESS!";
}

TEST(pool_arm, compute) {
  //   // speedup for ci
  //   for (auto pooling_type : {"max", "avg"}) {
  //     for (auto ceil_mode : {true, false}) {
  //       for (auto global_pooling : {true, false}) {
  //         for (auto exclusive : {true, false}) {
  //           for (auto ksize : {2, 3}) {
  //             for (auto stride : {1, 2}) {
  //               for (auto pad : {0, 1}) {
  //                 for (auto n : {1, 2}) {
  //                   for (auto c : {1, 3}) {
  // #if 1
  //                     for (auto h : {2, 3, 4, 11}) {
  //                       for (auto w : {2, 3, 4, 11}) {
  // #else
  //                     for (int h = 2; h < 25; h++) {
  //                       for (int w = 2; w < 25; w++) {
  // #endif
  //                         VLOG(3) << "n:" << n << " c:" << c << " h:" << h
  //                                 << " w:" << w << " ksize:" << ksize
  //                                 << " stride:" << stride << " pad:" << pad
  //                                 << " exclusive:" << exclusive
  //                                 << " global_pooling:" << global_pooling
  //                                 << " ceil_mode: " << ceil_mode
  //                                 << " pooling_type:" << pooling_type;
  //                          test_vulkan_pool2d_fp32_16( {{n,c,h,w}},
  //                               pooling_type,
  //                               global_pooling,
  //                               ksize,
  //                               {stride,stride},
  //                               {pad,pad},
  //                               exclusive,
  //                               ceil_mode);
  //                       }
  //                     }
  //                   }
  //                 }
  //               }
  //             }
  //           }
  //         }
  //       }
  //     }
  //   }
  if (true) {
    for (auto pooling_type : {"max", "avg"}) {
      for (auto ceil_mode : {false}) {
        for (auto global_pooling : {false, true}) {
          for (auto exclusive : {true}) {
            for (auto ksize : {2}) {
              for (auto stride : {1}) {
                for (auto pad : {0}) {
                  for (auto n : {2}) {
                    for (auto c : {1024}) {
#if 1
                      for (auto h : {2}) {
                        for (auto w : {2}) {
#else
                      for (int h = 2; h < 25; h++) {
                        for (int w = 2; w < 25; w++) {
#endif
                          LOG(INFO) << "n:" << n << " c:" << c << " h:" << h
                                    << " w:" << w << " ksize:" << ksize
                                    << " stride:" << stride << " pad:" << pad
                                    << " exclusive:" << exclusive
                                    << " global_pooling:" << global_pooling
                                    << " ceil_mode: " << ceil_mode
                                    << " pooling_type:" << pooling_type;
                          test_vulkan_pool2d_fp32_16({DDim({n, c, h, w})},
                                                     pooling_type,
                                                     global_pooling,
                                                     ksize,
                                                     {stride, stride},
                                                     {pad, pad},
                                                     exclusive,
                                                     ceil_mode);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  // speedup for ci
  if (false) {
    for (auto pooling_type : {"max", "avg"}) {
      for (auto ceil_mode : {false}) {
        for (auto global_pooling : {false, true}) {
          for (auto exclusive : {true}) {
            for (auto ksize : {2}) {
              for (auto stride : {1}) {
                for (auto pad : {0}) {
                  for (auto n : {1}) {
                    for (auto c : {5}) {
#if 1
                      for (auto h : {2}) {
                        for (auto w : {2}) {
#else
                      for (int h = 2; h < 25; h++) {
                        for (int w = 2; w < 25; w++) {
#endif
                          LOG(INFO) << "n:" << n << " c:" << c << " h:" << h
                                    << " w:" << w << " ksize:" << ksize
                                    << " stride:" << stride << " pad:" << pad
                                    << " exclusive:" << exclusive
                                    << " global_pooling:" << global_pooling
                                    << " ceil_mode: " << ceil_mode
                                    << " pooling_type:" << pooling_type;
                          test_vulkan_pool2d_fp32_16({DDim({n, c, h, w})},
                                                     pooling_type,
                                                     global_pooling,
                                                     ksize,
                                                     {stride, stride},
                                                     {pad, pad},
                                                     exclusive,
                                                     ceil_mode);
                        }
                      }
                    }
                  }
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
USE_LITE_KERNEL(pool2d, kVULKAN, kFloat, kNCHW, def);
