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
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lite/backends/vulkan/shader_hex.h"
#include "lite/backends/vulkan/shader_header.h"
#include "lite/core/tensor.h"
#include "lite/utils/cp_logging.h"

#ifdef LITE_WITH_VULKAN
#define VK_USE_PLATFORM_ANDROID_KHR 1
#include <vulkan/vulkan.h>

// #include <vulkan/vulkan_core.h>
// #include "device_info.h"

#endif

namespace paddle {
namespace lite {
namespace vulkan {

#ifdef LITE_WITH_VULKAN

VkResult init_device(const struct vk_device& vk_device_);
VkResult init_enumerate_device(const struct vk_device& vk_device_,
                               uint32_t gpu_count = 1);
void init_device_queue(const struct vk_device& vk_device_);
void init_shader(const struct vk_device& vk_device_);

typedef struct {
  VkImage image;
  VkDeviceMemory devicemem;
  VkImageView view;
} VulkanImage;

typedef struct {
  VkBuffer buf;
  VkDeviceMemory devicemem;
  void* mapped_ptr;
  size_t size;
} VulkanBuf;

typedef union {
  VulkanBuf* buf;
  VulkanImage* image;
} Mem;
enum class MemType {
  BUF = 0,  // VulkanBuf
  IMAGE,    // VulkanImage
};
enum class BufTYPE {
  NCHW = 0,
  HW4,
  H4W,
};
typedef struct {
  MemType type;
  Mem mem;
} VulkanMem;
class VulkanDevice {
 public:
  VulkanDevice();
  // explicit VulkanDevice(VkInstance vk_inst );
  virtual ~VulkanDevice();
  void CreateInstance();
  void InitInstanceExtName();
  void InitInstanceExt();
  void InitEnumerateDevice(uint32_t gpu_count = 1);
  void EnumeratePhysicalDevices(uint32_t gpu_count = 1);
  void GetPhysicalDeviceQueueFamilyProperties();
  void GetPhysicalDeviceMemoryProperties();
  void GetPhysicalDeviceProperties();
  void EnumerateDeviceExtensionProperties();
  void CreateDevice();
  void GetDeviceComputeQueue();
  void vkCreateShaders();
  void CreateSampler();
  void CreateFence();
  void CreatePipeLineCache();
  VulkanImage* CreateImage(VkImageType imageType,
                           VkImageViewType viewType,
                           uint32_t depth,
                           uint32_t height,
                           uint32_t width,
                           VkFormat format);
  // void CreateImage(VkImageType imageType, VkImageViewType viewType, uint32_t
  // depth, uint32_t height, uint32_t width, VkFormat format);
  VulkanBuf* CreateBuffer(uint32_t size);
  void MapMemory(VulkanBuf* bufdata,
                 const VkDeviceMemory memory,
                 const int start,
                 const int size);

  void UnMapMemory(const VulkanBuf* bufdata, const VkDeviceMemory memory);

  void FlushBuffer(VulkanBuf* buf, uint64_t start, uint64_t size);

  void AllocateMemmory(VkMemoryRequirements* memoryRequirements,
                       VkDeviceMemory* mem,
                       VkFlags memTpye);

  void CreateImageView(VkImageView* view,
                       const VkImage& image,
                       const VkImageViewType& type,
                       const VkFormat& format);

  uint32_t GetComputQueueIdx();
  VkDevice GetVkDevice();

  void CreateDescriptorSetLayout(const std::vector<VkDescriptorType>& types);
  void CreatePipelineLayout(const VkDescriptorSetLayout& setLayout);
  VkShaderModule GetShaderModule(const std::string& name);

  void CreateComputePipelines(
      const std::string& name,
      const std::vector<uint32_t>& localSize = std::vector<uint32_t>());

  // void CreateDescriptorPool(const VkDescriptorPoolCreateInfo* info);
  // void CreateDescriptorPool(const VkDescriptorType type, const uint32_t
  // des_count, const uint32_t set_count);
  void CreateDescriptorPool(const uint32_t set_count);

  void AllocateDescriptorSets(const VkDescriptorPool& descPool,
                              const VkDescriptorSetLayout& setLayout);
  // void UpdateDescriptorSets(VkDevice device, int size, VkBuffer buf, int
  // index);
  void UpdateDescriptorSets(VkDevice device,
                            int size,
                            VkBuffer buf,
                            int index,
                            VkDescriptorType type);
  void UpdateDescriptorSets(VkDevice device,
                            const VulkanMem* vulkanmem,
                            int index,
                            VkDescriptorType type,
                            VkImageLayout layout);

  void QueueSubmit(VkQueue compute_queue, VkCommandBuffer cmd, VkFence fence);
  void QueueSubmit(VkQueue compute_queue,
                   std::vector<VkCommandBuffer>* cmd,
                   VkFence fence);

  VkFence fence;
  VkQueue compute_queue;

  //--------------
  VkPipelineLayout pipelineLayout;
  VkDescriptorSetLayout setLayout;
  // VkPipelineCache pipelineCache;
  // VkRenderPass render_pass;
  VkDescriptorUpdateTemplateKHR descriptor_update_template;
  VkPipeline pipeline;

  // std::vector<VkDescriptorPool> descriptor_pools;
  // std::vector<VkDescriptorSet> descriptorsets;
  uint32_t local_size_x;
  uint32_t local_size_y;
  uint32_t local_size_z;

  std::vector<VkDescriptorPoolSize> pool_size;

  //--------------
  std::vector<VkShaderModule> shader_modules;

  std::map<const std::string, VkShaderModule> shader_modules_map;

  //--
  VkDescriptorPool descriptorpool[1] = {};
  VkDescriptorSet descriptorSet[1] = {};

  //--

  void CreateCommandPool();

  void AllocateCommandBuffers();
  void AllocateCommandBuffers(VkCommandBuffer* cmd);
  void BeginCommandBuffer(VkCommandBuffer cmd, VkCommandBufferUsageFlags flag);
  void EndCommandBuffer(VkCommandBuffer cmd);

  VkCommandBuffer cmd;                // Buffer for initialization commands
  VkCommandBuffer ttcmd;              // Buffer for initialization commands
  std::vector<VkCommandBuffer> cmds;  // Buffer for initialization commands

 private:
  bool prepared;
  bool use_staging_buffer;
  bool save_images;

  std::vector<const char*> instance_layer_names;
  std::vector<const char*> instance_extension_names;
  // std::vector<layer_properties> instance_layer_properties;

  VkLayerProperties properties;
  std::vector<VkExtensionProperties> instance_extensions;
  std::vector<VkExtensionProperties> device_extensions;
  // std::vector<VkExtensionProperties> instance_extension_properties;
  VkInstance inst;

  std::vector<const char*> device_extension_names;  // NULL  NULL?
  std::vector<VkExtensionProperties> device_extension_properties;
  std::vector<VkPhysicalDevice> gpus;
  VkDevice device;
  // VkQueue graphics_queue;
  // uint32_t graphics_queue_family_index;
  uint32_t compute_queue_family_index;
  VkPhysicalDeviceProperties gpu_props;
  std::vector<VkQueueFamilyProperties> queue_props;
  VkPhysicalDeviceMemoryProperties memory_properties;
  int device_local_idx;

  VkFramebuffer* framebuffers;
  // int width, height;
  VkFormat format;

  VkCommandPool cmdpool;

  // VkBuffer buf;

  std::vector<VulkanImage*> images;
  std::vector<VulkanBuf*> pbufs;

  uint32_t queue_family_count;
  VkSampler sampler;
  VkResult res;
  VkPipelineCache pipelineCache;

  //     VkImage image;
  // VkDeviceMemory mem;
  // VkImageView view;
  // void* mapped_ptr;
};
#endif
}  // namespace vulkan
}  // namespace lite
}  // namespace paddle
