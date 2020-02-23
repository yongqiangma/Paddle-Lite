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

#include "lite/kernels/vulkan/conv_compute.h"
#include <utility>
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#include "lite/kernels/vulkan/conv_depthwise.h"
#include "lite/kernels/vulkan/conv_direct.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace vulkan {

template <>
void ConvCompute<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  auto& param = this->Param<param_t>();
  auto w_dims = param.filter->dims();
  auto& ctx = this->ctx_->template As<VULKANContext>();

  int ic = w_dims[1] * param.groups;
  int oc = w_dims[0];
  int kh = w_dims[2];  // oihw
  int kw = w_dims[3];
  int pad = param.paddings->data()[0];
  int stride = param.strides[0];

  int chin = param.x->dims()[1];
  int hin = param.x->dims()[2];
  int win = param.x->dims()[3];
  int chout = param.output->dims()[1];
  int hout = param.output->dims()[2];
  int wout = param.output->dims()[3];

  bool kps_equal = (param.paddings->data()[0] == param.paddings->data()[1]) &&
                   (param.strides[0] == param.strides[1]) && (kw == kh);
  bool no_dilation =
      (param.dilations->data()[0] == 1) && (param.dilations->data()[1] == 1);
  bool flag_dw_3x3 = (kw == 3 && kh == 3 && (stride == 1 || stride == 2));
  bool flag_dw_5x5 =
      (kw == 5 && stride == 1) || (kw == 5 && stride == 2 && pad == 2);
  bool flag_dw = flag_dw_3x3 || flag_dw_5x5;

  /// select conv impl
  if (param.groups == ic) {
    /// dw conv impl
    impl_ = new DepthwiseConv();
    VLOG(3) << "invoking dw conv";
  } else if (param.groups == 1) {
    /// direct conv impl
    impl_ = new DirectConv;
    VLOG(3) << "invoking direct conv";
  } else {
    LOG(FATAL) << "vulkan don't support this conv";
  }
  impl_->SetContext(std::move(this->ctx_));
  impl_->SetParam(param);
  impl_->PrepareForRun();
  is_first_epoch_ = false;
}

}  // namespace vulkan
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

typedef paddle::lite::kernels::vulkan::ConvCompute<PRECISION(kFloat),
                                                   PRECISION(kFloat)>
    ConvFp32;

REGISTER_LITE_KERNEL(conv2d, kVULKAN, kFloat, kNCHW, ConvFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kVULKAN))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kVULKAN))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kVULKAN))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kVULKAN))})
    .Finalize();

REGISTER_LITE_KERNEL(depthwise_conv2d, kVULKAN, kFloat, kNCHW, ConvFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kVULKAN))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kVULKAN))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kVULKAN))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kVULKAN))})
    .Finalize();
