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

#include "lite/backends/vulkan/device_info.h"

namespace paddle {
namespace lite {
namespace vulkan {
#ifdef LITE_WITH_VULKAN

VkResult init_global_extension_properties(const vk_device& vk_device_) {
  VkExtensionProperties* instance_extensions;
  uint32_t instance_extension_count;
  VkResult res;
  char* layer_name = NULL;

  layer_name = vk_device_.properties.layerName;

  do {
    res = vkEnumerateInstanceExtensionProperties(
        layer_name, &instance_extension_count, NULL);

    if (res) return res;

    if (instance_extension_count == 0) {
      return VK_SUCCESS;
    }

    vk_device_.instance_extensions.resize(instance_extension_count);
    instance_extensions = vk_device_.instance_extensions.data();
    res = vkEnumerateInstanceExtensionProperties(
        layer_name, &instance_extension_count, instance_extensions);
  } while (res == VK_INCOMPLETE);

  return res;
}

void init_instance_extension_names(const vk_device& vk_device_) {
  vk_device_.instance_extension_names.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
#ifdef __ANDROID__
  vk_device_.instance_extension_names.push_back(
      VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
#elif defined(_WIN32)
  vk_device_.instance_extension_names.push_back(
      VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_IOS_MVK)
  vk_device_.instance_extension_names.push_back(
      VK_MVK_IOS_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_MACOS_MVK)
  vk_device_.instance_extension_names.push_back(
      VK_MVK_MACOS_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
  vk_device_.instance_extension_names.push_back(
      VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME);
#else
  vk_device_.instance_extension_names.push_back(
      VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#endif
}

VkResult init_instance(const vk_device& vk_device_) {
  VkApplicationInfo app_info = {};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pNext = NULL;
  app_info.pApplicationName = "paddle ite";
  app_info.applicationVersion = 1;
  app_info.pEngineName = "paddle lite";
  app_info.engineVersion = 1;
  app_info.apiVersion = VK_API_VERSION_1_0;

  VkInstanceCreateInfo inst_info = {};
  inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  inst_info.pNext = NULL;
  inst_info.flags = 0;
  inst_info.pApplicationInfo = &app_info;
  inst_info.enabledLayerCount = 0;
  inst_info.ppEnabledLayerNames = NULL;
  inst_info.enabledExtensionCount = vk_device_.instance_extension_names.size();
  inst_info.ppEnabledExtensionNames =
      vk_device_.instance_extension_names.data();

  VkResult res = vkCreateInstance(&inst_info, NULL, &vk_device_.inst);
  assert(res == VK_SUCCESS);

  return res;
}

VkResult init_device_extension_properties(const vk_device& vk_device_,
                                          const layer_properties& layer_props) {
  VkExtensionProperties* device_extensions;
  uint32_t device_extension_count;
  VkResult res;
  char* layer_name = NULL;

  layer_name = layer_props.properties.layerName;

  do {
    res = vkEnumerateDeviceExtensionProperties(
        vk_device_.gpus[0], layer_name, &device_extension_count, NULL);
    if (res) return res;

    if (device_extension_count == 0) {
      return VK_SUCCESS;
    }

    layer_props.device_extensions.resize(device_extension_count);
    device_extensions = layer_props.device_extensions.data();
    res = vkEnumerateDeviceExtensionProperties(vk_device_.gpus[0],
                                               layer_name,
                                               &device_extension_count,
                                               device_extensions);
    LOG(INFO) << "layer_props.device_extensions"
              << device_extensions->extensionName;
  } while (res == VK_INCOMPLETE);

  return res;
}

#endif  // LITE_WITH_VULKAN
}  // namespace vulkan
}  // namespace lite
}  // namespace paddle
