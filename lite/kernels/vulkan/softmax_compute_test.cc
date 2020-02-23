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

template <typename dtype>
void softmax_compute_ref(const operators::SoftmaxParam& param) {
  const dtype* x_data = param.x->mutable_data<const dtype>();
  dtype* output_data = param.output->mutable_data<dtype>();
  DDim x_dims = param.x->dims();
  ASSERT_EQ(x_dims.data(), param.output->dims().data());
  auto x_rank = x_dims.size();
  int axis = param.axis;
  if (axis < 0) {
    axis += x_rank;
  }
  int axis_size = x_dims[axis];
  int outer_num = x_dims.Slice(0, axis).production();
  int inner_num = x_dims.Slice(axis + 1, x_rank).production();
  int compute_size = outer_num * inner_num;
  for (int i = 0; i < compute_size; i++) {
    int idx_inner = i % inner_num;
    int idx_outer = (i / inner_num) * axis_size;
    int start = idx_outer * inner_num + idx_inner;
    int offset;

    offset = start;
    dtype max_data = std::numeric_limits<dtype>::lowest();
    for (int j = 0; j < axis_size; j++) {
      max_data = x_data[offset] > max_data ? x_data[offset] : max_data;
      offset += inner_num;
    }
    // LOG(INFO)<<"max_data:"<<max_data<<"  offset:"<<offset<<"
    // inner_num:"<<inner_num<<"   outer_num:"<<outer_num;
    offset = start;
    dtype sum_data = (dtype)0;
    for (int j = 0; j < axis_size; j++) {
      output_data[offset] = exp(x_data[offset] - max_data);
      LOG(INFO) << "x_data[offset] - max_data:" << x_data[offset] - max_data;

      // LOG(INFO)<<"x_data[offset] - max_data:" << x_data[offset] - max_data
      // <<"max_data:"<<max_data<<"  offset:"<<offset<<"
      // inner_num:"<<inner_num<<"   outer_num:"<<outer_num<<"
      // data:"<<output_data[offset];
      sum_data += output_data[offset];
      offset += inner_num;
    }
    LOG(INFO) << "sum_data:" << sum_data;
    offset = start;
    for (int j = 0; j < axis_size; j++) {
      output_data[offset] /= sum_data;
      offset += inner_num;
    }
  }
}

void test_vulkan_softmax(
    const int n, const int c, const int h, const int w, const int axis) {
  LOG(INFO) << "vulkan_fc test";

  auto kernels = KernelRegistry::Global().Create(
      "softmax", TARGET(kVULKAN), PRECISION(kFloat), DATALAYOUT(kNCHW));

  ASSERT_FALSE(kernels.empty());

  auto kernel = std::move(kernels.front());

  auto context = ContextScheduler::Global().NewContext(TARGET(kVULKAN));

  lite::Tensor input, input_vk, output, weight, weight_vk, bias, bias_vk,
      input_im, out_buf, out_buf_vk, output_ref;

  auto device = context->As<VULKANContext>().device();

  operators::SoftmaxParam param;
  // fill param

  param.x = &input_vk;
  param.output = &output;
  param.axis = axis;

  kernel->SetParam(param);

  std::unique_ptr<KernelContext> dep_context(new KernelContext);
  context->As<VULKANContext>().CopySharedTo(
      &(dep_context->As<VULKANContext>()));
  kernel->SetContext(std::move(dep_context));

  input.Resize(DDim(std::vector<int64_t>({n, c, h, w})));
  output.Resize(DDim(std::vector<int64_t>({n, c, h, w})));
  output_ref.Resize(DDim(std::vector<int64_t>({n, c, h, w})));
  auto* x_data = input.mutable_data<float>();
  auto* output_data = output.mutable_data<float>();

  /*
   x_data = {{{2.0, 3.0, 4.0, 5.0},
              {3.0, 4.0, 5.0, 6.0},
              {7.0, 8.0, 8.0, 9.0}},
             {{1.0, 2.0, 3.0, 4.0},
              {5.0, 6.0, 7.0, 8.0},
              {6.0, 7.0, 8.0, 9.0}}};
   */
  for (int i = 0; i < input.dims().production(); i++) {
    x_data[i] = (i + 1) / 5;
    LOG(INFO) << "in:" << x_data[i];
  }

  CopyToVulkan(device, &input, &input_vk);

  kernel->Launch();

  int FLAGS_repeats = 5;
  auto start = GetCurrentUS();
  for (int i = 0; i < FLAGS_repeats; ++i) {
    VulkanRun(device);
  }
  auto time = (GetCurrentUS() - start) / FLAGS_repeats / 1000.0;
  LOG(INFO) << "H:" << output.dims() << "time:" << time;

  CopyToHost(device, &output, &out_buf);
  auto* out_cpu = out_buf.data<float>();

  auto output_ref_data = output_ref.mutable_data<float>();

  param.x = &input;
  param.output = &output_ref;
  softmax_compute_ref<float>(param);
  // compare
  for (int i = 0; i < output.dims().production(); i++) {
    LOG(INFO) << i << " out:" << out_cpu[i] << "ref:" << output_ref_data[i];
    // LOG(INFO) << i <<  "ref:" << output_ref_data[i];
  }
  for (int i = 0; i < output.dims().production(); i++) {
    float gap = output_ref_data[i] * 2 / 1000;
    gap = gap < 1e-4 ? 1e-4 : gap;
    EXPECT_NEAR(out_cpu[i], output_ref_data[i], gap);
    // LOG(INFO)<<"  out:"<<out_cpu[i]<<"ref:"<<output_ref_data[i];
  }
  LOG(INFO) << "SUCCESS!";
}
TEST(softmax_vulkan, compute) {
#if 1
  for (auto n : {1}) {
    for (auto c : {1000}) {
      for (auto h : {1}) {
        for (auto w : {1}) {
          for (auto axis : {1}) {
#else
  for (auto n : {1, 3, 4, 11}) {
    for (auto c : {1, 3, 11, 4}) {
      for (auto h : {3, 1, 11, 4}) {
        for (auto w : {1, 3, 4, 12}) {
          for (auto axis : {-4, -3, -2, -1, 0, 1, 2, 3}) {
#endif
            test_vulkan_softmax(n, c, h, w, axis);
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
USE_LITE_KERNEL(softmax, kVULKAN, kFloat, kNCHW, def);
