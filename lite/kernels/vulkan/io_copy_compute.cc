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

#include "lite/api/paddle_place.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/target_wrapper.h"
#include "lite/core/type_system.h"
#include "lite/kernels/vulkan/vulkan_utils.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace vulkan {
/*
 * This kernel copies a tensor from host to GPU space.
 */
class IoCopyHostToVULKANCompute
    : public KernelLite<TARGET(kVULKAN), PRECISION(kFloat), DATALAYOUT(kNCHW)> {
 public:
  void Run() override {
    auto& param = Param<operators::IoCopyParam>();
    CHECK(param.x->target() == TARGET(kHost) ||
          param.x->target() == TARGET(kVULKAN));
    auto context = ContextScheduler::Global().NewContext(TARGET(kVULKAN));

    auto device = context->As<VULKANContext>().device();
    /*
    LOG(INFO)<<"IoCopyHostToVULKANCompute:"<<param.x->dims();
    if(param.x->dims().production() == 32*3*3*3 ||param.x->dims().production()
    == 32)
    {
      auto* i_data = param.x->data<float>();
        LOG(INFO)<<"conv_direct_xio dims"<<param.x->dims();
    for(int i = 0; i < param.x->dims().production(); i++){
      //LOG(INFO)<<"conv_direct_xio:"<<i << " :"<<i_data[i];
    }
    }
  */

    CopyToVulkan(device, param.x, param.y);
  }

  std::string doc() const override { return "Copy IO from HOST to VULKAN_GPU"; }
};

/*
 * This kernel copies a tensor from GPU to host space.
 */
class IoCopyVULKANToHostCompute
    : public KernelLite<TARGET(kVULKAN), PRECISION(kFloat), DATALAYOUT(kNCHW)> {
 public:
  void Run() override {
    auto& param = Param<operators::IoCopyParam>();
    CHECK(param.x->target() == TARGET(kHost) ||
          param.x->target() == TARGET(kVULKAN));

    auto context = ContextScheduler::Global().NewContext(TARGET(kVULKAN));

    auto device = context->As<VULKANContext>().device();
    VulkanRun(device);
    //     LOG(INFO)<<"copytohost_dim x:"<<param.x->dims();
    //    VulkanImage2Buf(device, param.x, &t_vk, BufTYPE::NCHW);

    CopyToHost(device, param.x, param.y);
    //    PrintTensor(param.y,"io_copy");
    //     LOG(INFO)<<"copytohost_dim y:"<<param.y->dims();
  }

  std::string doc() const override { return "Copy IO from VULKAN_GPU to HOST"; }
};

}  // namespace vulkan
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(io_copy,
                     kVULKAN,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::vulkan::IoCopyHostToVULKANCompute,
                     host_to_device)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kVULKAN),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(io_copy,
                     kVULKAN,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::vulkan::IoCopyVULKANToHostCompute,
                     device_to_host)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kVULKAN),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(io_copy_once,
                     kVULKAN,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::vulkan::IoCopyHostToVULKANCompute,
                     host_to_device_once)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kVULKAN),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

REGISTER_LITE_KERNEL(io_copy_once,
                     kVULKAN,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::vulkan::IoCopyVULKANToHostCompute,
                     device_to_host_once)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kVULKAN),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kARM),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
