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

#include <cstdarg>
#include <memory>
#include <string>
#include <vector>
#include "lite/core/tensor.h"
#include "lite/utils/cp_logging.h"

#ifdef LITE_WITH_VULKAN
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_core.h>
#include "lite/backends/vulkan/vk_device.h"
#endif

namespace paddle {
namespace lite {
namespace vulkan {

#ifdef LITE_WITH_VULKAN

void init_command_pool(const struct vk_device &vk_device_);

class VulkanCommand {
 public:
  VulkanCommand();
  explicit VulkanCommand(std::shared_ptr<VulkanDevice> dev);
  void Init(std::shared_ptr<VulkanDevice> dev);
  void Create(VkCommandPool *_pool);
  ~VulkanCommand();
  // private:
  std::shared_ptr<VulkanDevice> vk_dev;
  VkCommandPool cmd_pool;
};
#endif

}  // namespace vulkan
}  // namespace lite
}  // namespace paddle
