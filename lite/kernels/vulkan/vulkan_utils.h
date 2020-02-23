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

#include "lite/backends/vulkan/vk_device.h"
#include "lite/core/kernel.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace vulkan {
using BufTYPE = lite::vulkan::BufTYPE;
extern DDim InitImageDimInfoWith(const DDim& tensor_dim);
extern DDim H4WInitImageDimInfoWith(const DDim& tensor_dim);
extern DDim HW4InitImageDimInfoWith(const DDim& tensor_dim);
extern void CopyToVulkan(std::shared_ptr<lite::vulkan::VulkanDevice> device,
                         const lite::Tensor* host,
                         lite::Tensor* vk);
extern void CopyToHost(std::shared_ptr<lite::vulkan::VulkanDevice> device,
                       const lite::Tensor* vk,
                       lite::Tensor* host);
extern void CopyToHost(std::shared_ptr<lite::vulkan::VulkanDevice> device,
                       const lite::Tensor* vk,
                       lite::Tensor* host);

extern void PrintTensor(const lite::Tensor* in, const std::string name);
extern void VulkanMatrixTrans(
    std::shared_ptr<lite::vulkan::VulkanDevice> device,
    const lite::Tensor* input,
    lite::Tensor* output);
extern void VulkanBuf2Image(std::shared_ptr<lite::vulkan::VulkanDevice> device,
                            const lite::Tensor* input,
                            lite::Tensor* output,
                            BufTYPE type);
extern void VulkanImage2Buf(std::shared_ptr<lite::vulkan::VulkanDevice> device,
                            const lite::Tensor* image,
                            lite::Tensor* output,
                            BufTYPE type);

extern void VulkanPrintImage(std::shared_ptr<lite::vulkan::VulkanDevice> device,
                             const lite::Tensor* image,
                             BufTYPE type);

void VulkanRun(std::shared_ptr<lite::vulkan::VulkanDevice> device);
}  // namespace vulkan
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
