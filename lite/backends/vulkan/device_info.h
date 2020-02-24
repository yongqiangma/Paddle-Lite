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
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lite/core/tensor.h"
#include "lite/utils/cp_logging.h"

#ifdef LITE_WITH_VULKAN
#define VK_USE_PLATFORM_ANDROID_KHR 1
#include <vulkan/vulkan.h>
// #include <vulkan/vulkan_core.h>

#include "lite/backends/vulkan/vk_command.h"
#include "lite/backends/vulkan/vk_device.h"
#endif

namespace paddle {
namespace lite {
namespace vulkan {

#ifdef LITE_WITH_VULKAN

struct texture_object {
  VkSampler sampler;

  VkImage image;
  VkImageLayout imageLayout;

  bool needs_staging;
  VkBuffer buffer;
  VkDeviceSize buffer_size;

  VkDeviceMemory image_memory;
  VkDeviceMemory buffer_memory;
  VkImageView view;
  int32_t tex_width, tex_height;
};

typedef struct _swap_chain_buffers {
  VkImage image;
  VkImageView view;
} swap_chain_buffer;

typedef struct {
  VkLayerProperties properties;
  std::vector<VkExtensionProperties> instance_extensions;
  std::vector<VkExtensionProperties> device_extensions;
} layer_properties;

struct vk_device {
  // #ifdef _WIN32
  // #define APP_NAME_STR_LEN 80
  //     HINSTANCE connection;        // hInstance - Windows Instance
  //     char name[APP_NAME_STR_LEN]; // Name to put on the window/icon
  //     HWND window;                 // hWnd - window handle
  // #elif (defined(VK_USE_PLATFORM_IOS_MVK) ||
  // defined(VK_USE_PLATFORM_MACOS_MVK))
  //    void* window;
  // #elif defined(__ANDROID__)
  //     PFN_vkCreateAndroidSurfaceKHR fpCreateAndroidSurfaceKHR;
  // #elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
  //     wl_display *display;
  //     wl_registry *registry;
  //     wl_compositor *compositor;
  //     wl_surface *window;
  //     wl_shell *shell;
  //     wl_shell_surface *shell_surface;
  // #else
  //     xcb_connection_t *connection;
  //     xcb_screen_t *screen;
  //     xcb_window_t window;
  //     xcb_intern_atom_reply_t *atom_wm_delete_window;
  // #endif // _WIN32
  VkSurfaceKHR surface;
  bool prepared;
  bool use_staging_buffer;
  bool save_images;

  std::vector<const char *> instance_layer_names;
  std::vector<const char *> instance_extension_names;

  VkLayerProperties properties;
  std::vector<VkExtensionProperties> instance_extensions;
  std::vector<VkExtensionProperties> device_extensions;
  VkInstance inst;

  std::vector<const char *> device_extension_names;  // NULL  NULL?
  std::vector<VkExtensionProperties> device_extension_properties;
  std::vector<VkPhysicalDevice> gpus;
  VkDevice device;
  // VkQueue graphics_queue;
  VkQueue compute_queue;
  VkFence fence;
  // uint32_t graphics_queue_family_index;
  uint32_t compute_queue_family_index;
  VkPhysicalDeviceProperties gpu_props;
  std::vector<VkQueueFamilyProperties> queue_props;
  VkPhysicalDeviceMemoryProperties memory_properties;
  int device_local_idx;

  VkFramebuffer *framebuffers;
  VkFormat format;
  VkBuffer buf;
  int size;
  VkDeviceMemory mem;
  void *mapped_ptr;
  // uint32_t current_buffer;
  uint32_t queue_family_count;
  std::vector<VkShaderModule> shader_modules;
};
typedef vk_device vk_device;

}  // namespace vulkan
}  // namespace lite
}  // namespace paddle
