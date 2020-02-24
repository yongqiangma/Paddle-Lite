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

// Parts of the following code in this file refs to
// https://github.com/Tencent/ncnn/blob/master/src/cpu.cpp
// Tencent is pleased to support the open source community by making ncnn
// available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the
// License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#include <algorithm>
#include <limits>

#include "lite/backends/vulkan/vk_command.h"
namespace paddle {
namespace lite {
namespace vulkan {
#ifdef LITE_WITH_VULKAN
VulkanCommand::VulkanCommand() {}

VulkanCommand::VulkanCommand(std::shared_ptr<VulkanDevice> dev) : vk_dev(dev) {
  VkCommandPool _pool;
  //             VkCommandPoolCreateInfo cmd_pool_info = {};
  //     cmd_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  //     cmd_pool_info.pNext = nullptr;
  //     cmd_pool_info.queueFamilyIndex = vk_dev->GetComputQueueIdx();
  //     cmd_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  // LOG(INFO) << "===============vkCreateCommandPool";

  //     VkResult res = vkCreateCommandPool(device, &cmd_pool_info, NULL,
  //     &_pool);
  //     if(res) LOG(FATAL)<<"CommandPool Create error!";
}
void VulkanCommand::Init(std::shared_ptr<VulkanDevice> dev) {
  vk_dev = dev;
  if (dev == VK_NULL_HANDLE) LOG(INFO) << "VK_NULL_HANDLE! index:";
  // VkCommandPoolCreateInfo cmd_pool_info = {};
  // cmd_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  // cmd_pool_info.pNext = NULL;
  // cmd_pool_info.queueFamilyIndex = vk_dev->GetComputQueueIdx();
  // cmd_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  //  LOG(INFO)<<"vkCreateCommandPool! index:"<<vk_dev->GetComputQueueIdx();

  // VkResult res = vkCreateCommandPool(vk_dev->GetVkDevice(), &cmd_pool_info,
  // NULL, _pool);
  // if(!res) LOG(FATAL)<<"CommandPool Create error!";
}
void VulkanCommand::Create(VkCommandPool* _pool) {
  VkCommandPoolCreateInfo cmd_pool_info = {};
  cmd_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  cmd_pool_info.pNext = NULL;
  cmd_pool_info.queueFamilyIndex = vk_dev->GetComputQueueIdx();
  cmd_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  LOG(INFO) << "===============vkCreateCommandPool";
  VkResult res =
      vkCreateCommandPool(vk_dev->GetVkDevice(), &cmd_pool_info, NULL, _pool);
  if (res) LOG(FATAL) << "CommandPool Create error!";
}

VulkanCommand::~VulkanCommand() {}
// void init_command_pool(struct vk_device &vk_device_) {
//     /* DEPENDS on init_swapchain_extension() */
//     VkResult res;

//     VkCommandPoolCreateInfo cmd_pool_info = {};
//     cmd_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
//     cmd_pool_info.pNext = NULL;
//     cmd_pool_info.queueFamilyIndex = vk_device_.compute_queue_family_index;
//     cmd_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

//     res = vkCreateCommandPool(vk_device_.device, &cmd_pool_info, NULL,
//     &vk_device_.cmd_pool);
//     assert(res == VK_SUCCESS);
// }

// void execute_begin_command_buffer(struct vk_device &vk_device_) {
//     /* DEPENDS on init_command_buffer() */
//     VkResult  res;

//     VkCommandBufferBeginInfo cmd_buf_info = {};
//     cmd_buf_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
//     cmd_buf_info.pNext = NULL;
//     cmd_buf_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;//??
//     cmd_buf_info.pInheritanceInfo = NULL;

//     res = vkBeginCommandBuffer(vk_device_.cmd, &cmd_buf_info);
//     assert(res != VK_SUCCESS);
// }

// void init_command_buffer(struct vk_device &vk_device_) {
//     /* DEPENDS on init_swapchain_extension() and init_command_pool() */
//     VkResult res;

//     VkCommandBufferAllocateInfo cmd = {};
//     cmd.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
//     cmd.pNext = NULL;
//     cmd.commandPool = vk_device_.cmd_pool;
//     cmd.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
//     cmd.commandBufferCount = 1;

//     res = vkAllocateCommandBuffers(vk_device_.device, &cmd, &vk_device_.cmd);
//     assert(res != VK_SUCCESS);

//     execute_begin_command_buffer(vk_device_);
// }
#endif  // LITE_WITH_VULKAN
}  // namespace vulkan
}  // namespace lite
}  // namespace paddle
