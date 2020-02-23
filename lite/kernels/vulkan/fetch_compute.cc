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

#include "lite/kernels/vulkan/fetch_compute.h"
#include "lite/core/op_registry.h"

#include "lite/kernels/vulkan/vulkan_utils.h"
namespace paddle {
namespace lite {
namespace kernels {
namespace vulkan {

void FetchCompute::Run() {
  LOG(INFO) << "vulkan_fetch rundddddddddddd";
  auto& param = Param<operators::FetchParam>();
  auto* fetch_list = param.fetch_list;
  if (fetch_list->size() <= static_cast<size_t>(param.col)) {
    fetch_list->resize(param.col + 1);
  }

  auto& dst = fetch_list->at(param.col);

  auto context = ContextScheduler::Global().NewContext(TARGET(kVULKAN));

  auto device = context->As<VULKANContext>().device();
  VulkanRun(device);
  // LOG(INFO)<<"copytohost_dim x:"<<param.x->dims();
  //    VulkanImage2Buf(device, param.x, &t_vk, BufTYPE::NCHW);

  // CopyToHost(device, param.x,param.y);
  CopyToHost(device, param.input, &dst);
  PrintTensor(&dst, "fetch");
}
/*
  auto& param = Param<operators::FetchParam>();
  auto* fetch_list = param.fetch_list;
  if (fetch_list->size() <= static_cast<size_t>(param.col)) {
    fetch_list->resize(param.col + 1);
  }
    auto context = ContextScheduler::Global().NewContext(TARGET(kVULKAN));

    auto device = context->As<VULKANContext>().device();
     VulkanRun(device);
    // LOG(INFO)<<"copytohost_dim x:"<<param.x->dims();
//    VulkanImage2Buf(device, param.x, &t_vk, BufTYPE::NCHW);

    CopyToHost(device, param.x,param.y);
    PrintTensor(param.x,"io_copy");
*/
}  // namespace vulkan
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(fetch,
                     kVULKAN,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::vulkan::FetchCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kVULKAN),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(
                    TARGET(kHost), PRECISION(kAny), DATALAYOUT(kAny), -1)})
    .Finalize();
