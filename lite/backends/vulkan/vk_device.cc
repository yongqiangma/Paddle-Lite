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

#include "lite/backends/vulkan/vk_device.h"

namespace paddle {
namespace lite {
namespace vulkan {
#ifdef LITE_WITH_VULKAN

VulkanDevice::VulkanDevice() {
  // InitInstanceExt();
  uint32_t layerCount;
  vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
  LOG(INFO) << "layerCount:" << layerCount;
  CreateInstance();
  InitEnumerateDevice(1);
  CreateDevice();
  GetDeviceComputeQueue();
  vkCreateShaders();
  CreateSampler();
  CreateFence();
  CreatePipeLineCache();
}

void VulkanDevice::InitInstanceExt() {
  uint32_t instance_extension_count;
  char* layer_name = NULL;
  layer_name = properties.layerName;
  do {
    res = vkEnumerateInstanceExtensionProperties(
        layer_name, &instance_extension_count, NULL);
    if (res || (instance_extension_count == 0)) {
      LOG(FATAL) << "Vulkan Instance Extension Init error!";
    }

    instance_extensions.resize(instance_extension_count);
    res = vkEnumerateInstanceExtensionProperties(
        layer_name, &instance_extension_count, instance_extensions.data());
    if (res) {
      LOG(FATAL) << "Vulkan Instance Extension Init error!";
    }
    LOG(INFO) << "layer_name:---------" << layer_name;
  } while (res == VK_INCOMPLETE);
}

void VulkanDevice::InitInstanceExtName() {
  instance_extension_names.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
#ifdef __ANDROID__
  instance_extension_names.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
#elif defined(_WIN32)
  instance_extension_names.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_IOS_MVK)
  instance_extension_names.push_back(VK_MVK_IOS_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_MACOS_MVK)
  instance_extension_names.push_back(VK_MVK_MACOS_SURFACE_EXTENSION_NAME);
#elif defined(VK_USE_PLATFORM_WAYLAND_KHR)
  instance_extension_names.push_back(VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME);
#else
  instance_extension_names.push_back(VK_KHR_XCB_SURFACE_EXTENSION_NAME);
#endif
}

void VulkanDevice::CreateInstance() {
  InitInstanceExtName();
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
  inst_info.enabledExtensionCount = instance_extension_names.size();
  inst_info.ppEnabledExtensionNames = instance_extension_names.data();
  VLOG(5) << "vkCreateInstance";

  res = vkCreateInstance(&inst_info, NULL, &inst);
  if (res) LOG(FATAL) << "Vulkan Instance Create error!:" << res;
}

void VulkanDevice::InitEnumerateDevice(uint32_t gpu_count) {
  EnumeratePhysicalDevices(gpu_count);
  GetPhysicalDeviceQueueFamilyProperties();

  for (int i = 0; i < queue_family_count; i++) {
    if ((queue_props[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
      compute_queue_family_index = i;
      VLOG(5) << "===============compute_queue_family_index :"
              << compute_queue_family_index;
    }
  }
  VLOG(5) << "===============compute_queue_family_index :"
          << compute_queue_family_index;

  GetPhysicalDeviceMemoryProperties();
  GetPhysicalDeviceProperties();
  // EnumerateDeviceExtensionProperties();
}

void VulkanDevice::EnumeratePhysicalDevices(uint32_t gpu_count) {
  uint32_t const req_count = gpu_count;
  res = vkEnumeratePhysicalDevices(inst, &gpu_count, NULL);
  if (res || (gpu_count == 0)) {
    LOG(FATAL) << "Vulkan Enumerate Physical Devices  error!";
  }
  gpus.resize(gpu_count);
  VLOG(5) << "vkEnumeratePhysicalDevices";

  res = vkEnumeratePhysicalDevices(inst, &gpu_count, gpus.data());
  if (res && gpu_count >= req_count) {
    LOG(FATAL) << "Vulkan Enumerate Physical Devices  error!";
  }
}

void VulkanDevice::GetPhysicalDeviceQueueFamilyProperties() {
  vkGetPhysicalDeviceQueueFamilyProperties(gpus[0], &queue_family_count, NULL);
  if (queue_family_count < 1) {
    LOG(FATAL) << "Vulkan Get Physical Device Queue Family Properties error!";
  }

  queue_props.resize(queue_family_count);
  VLOG(5) << "vkGetPhysicalDeviceQueueFamilyProperties  "
             "queue_family_count:"
          << queue_family_count;

  vkGetPhysicalDeviceQueueFamilyProperties(
      gpus[0], &queue_family_count, queue_props.data());
  if (queue_family_count < 1) {
    LOG(FATAL) << "VulkanGet Physical Device Queue Family Properties error!";
  }
}

void VulkanDevice::GetPhysicalDeviceMemoryProperties() {
  VLOG(5) << "vkGetPhysicalDeviceMemoryProperties";

  vkGetPhysicalDeviceMemoryProperties(
      gpus[0], &memory_properties);  // should add host memory pointer

  for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
    const VkMemoryType& memoryType = memory_properties.memoryTypes[i];

    if (memoryType.propertyFlags == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
      device_local_idx = i;
      VLOG(5) << "device_local_idx:" << device_local_idx;
    }
  }
}

void VulkanDevice::GetPhysicalDeviceProperties() {
  VLOG(5) << "vkGetPhysicalDeviceProperties";

  vkGetPhysicalDeviceProperties(gpus[0], &gpu_props);
  VLOG(5) << "apiVsersion:" << VK_VERSION_MAJOR(gpu_props.apiVersion) << "."
          << VK_VERSION_MINOR(gpu_props.apiVersion) << "."
          << VK_VERSION_PATCH(gpu_props.apiVersion);
  VLOG(5) << "driverVersion:" << VK_VERSION_MAJOR(gpu_props.driverVersion)
          << "." << VK_VERSION_MINOR(gpu_props.driverVersion) << "."
          << VK_VERSION_PATCH(gpu_props.driverVersion);
  VLOG(5) << "vendorID:" << gpu_props.vendorID;
  VLOG(5) << "deviceID:" << gpu_props.deviceID;
  VLOG(5) << "deviceType:" << gpu_props.deviceType;
  VLOG(5) << "pipelineCacheUUID:" << gpu_props.pipelineCacheUUID;
  if (gpu_props.vendorID == 0x13b5) VLOG(5) << "gpu type is Mali";
  if (gpu_props.vendorID == 0x5143) VLOG(5) << "gpu type is Adreno";
}

void VulkanDevice::EnumerateDeviceExtensionProperties() {
  uint32_t device_extension_count = 0;
  res = vkEnumerateDeviceExtensionProperties(
      gpus[0], NULL, &device_extension_count, NULL);
  if (res || (device_extension_count == 0)) {
    LOG(FATAL) << "Vulkan Enumerate Device Extension Properties error!";
  }

  device_extensions.resize(device_extension_count);
  res = vkEnumerateDeviceExtensionProperties(
      gpus[0], NULL, &device_extension_count, device_extensions.data());
  if (res) {
    LOG(FATAL) << "Vulkan Enumerate Device Extension Properties error!";
  }
}

void VulkanDevice::CreateDevice() {
  VkDeviceQueueCreateInfo queue_info = {};

  float queue_priorities[1] = {0.0};
  queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_info.pNext = NULL;
  queue_info.queueCount = 1;
  queue_info.pQueuePriorities = queue_priorities;
  queue_info.queueFamilyIndex = compute_queue_family_index;

  VkDeviceCreateInfo device_info = {};
  device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_info.pNext = NULL;
  device_info.queueCreateInfoCount = 1;
  device_info.pQueueCreateInfos = &queue_info;
  device_info.enabledExtensionCount = device_extension_names.size();
  device_info.ppEnabledExtensionNames =
      device_info.enabledExtensionCount ? device_extension_names.data() : NULL;
  device_info.pEnabledFeatures = NULL;
  VLOG(5) << "vkCreateDevice";

  res = vkCreateDevice(gpus[0], &device_info, NULL, &device);
  if (res) {
    LOG(FATAL) << "Vulkan Device Create error!";
  }
}

void VulkanDevice::GetDeviceComputeQueue() {
  VLOG(5) << "vkGetDeviceQueue";

  vkGetDeviceQueue(device, compute_queue_family_index, 0, &compute_queue);
}

int shader_num = sizeof(vulkan_shaders) / sizeof(vulkan_shader);

void VulkanDevice::vkCreateShaders() {
  // shader_modules.resize(shader_num, VK_NULL_HANDLE);
  VLOG(5) << "shader_num:" << shader_num;
  for (int i = 0; i < shader_num; i++) {
    const char* shader_name = vulkan_shaders[i].name;

    VkShaderModuleCreateInfo shaderModuleCreateInfo;
    shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.pNext = 0;
    shaderModuleCreateInfo.flags = 0;
    shaderModuleCreateInfo.codeSize = vulkan_shaders[i].size;
    shaderModuleCreateInfo.pCode = vulkan_shaders[i].data;

    VkShaderModule shader_module;

    res = vkCreateShaderModule(
        device, &shaderModuleCreateInfo, 0, &shader_module);
    if (res)
      LOG(FATAL) << "Vulkan  Create Shader error:" << vulkan_shaders[i].name;
    // shader_modules[i] = shader_module;
    shader_modules_map.insert(
        std::make_pair(vulkan_shaders[i].name, shader_module));
    VLOG(5) << "create shader_module: " << shader_name;
  }
}

uint32_t VulkanDevice::GetComputQueueIdx() {
  return compute_queue_family_index;
}
VkDevice VulkanDevice::GetVkDevice() { return device; }
void VulkanDevice::CreateSampler() {
  VkSamplerCreateInfo info;
  ::memset(&info, 0, sizeof(info));
  info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  info.magFilter = VK_FILTER_NEAREST;
  info.minFilter = VK_FILTER_NEAREST;
  info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER;
  info.mipLodBias = 0.0f;
  info.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
  info.anisotropyEnable = VK_FALSE;
  info.maxAnisotropy = 1.0f;
  info.compareEnable = VK_FALSE;
  info.minLod = 0.0f;
  info.maxLod = 0.0f;
  VLOG(5) << "vkCreateSampler !!!!";

  res = vkCreateSampler(device, &info, nullptr, &sampler);
  if (res) LOG(FATAL) << "Vulkan Sampler Create error!";
}
void VulkanDevice::CreateFence() {
  VkFenceCreateInfo info;
  info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  info.pNext = nullptr;
  info.flags = 0;
  VLOG(5) << "vkCreateFence !!!!";

  res = vkCreateFence(device, &info, nullptr, &fence);
  if (res) LOG(FATAL) << "Vulkan  Create Fence error!";
}
void VulkanDevice::CreatePipeLineCache() {
  VkPipelineCacheCreateInfo info;
  info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
  info.pNext = nullptr;
  info.initialDataSize = 0;
  info.pInitialData = nullptr;
  info.flags = 0;
  VLOG(5) << "vkCreatePipelineCache !!!!";
  res = vkCreatePipelineCache(device, &info, nullptr, &pipelineCache);
  if (res) LOG(FATAL) << "Vulkan Create  PipelineCache error!";
}

VulkanImage* VulkanDevice::CreateImage(VkImageType imageType,
                                       VkImageViewType viewType,
                                       uint32_t depth,
                                       uint32_t width,
                                       uint32_t height,
                                       VkFormat format) {
  VulkanImage* im = new VulkanImage();

  VkImageCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  info.imageType = imageType;
  info.extent.width = width;
  info.extent.height = height;
  info.extent.depth = depth;
  info.mipLevels = 1;
  info.arrayLayers = 1;
  info.format = format;
  info.tiling = VK_IMAGE_TILING_OPTIMAL;
  info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  info.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
               VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  info.samples = VK_SAMPLE_COUNT_1_BIT;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  info.pNext = nullptr;
  VLOG(5) << "vkCreateImage !!!!";

  res = vkCreateImage(device, &info, nullptr, &im->image);
  if (res) LOG(FATAL) << "Vulkan Create Image error!";

  VkMemoryRequirements memRequirements;
  VLOG(5) << "vkGetImageMemoryRequirements !!!!";
  vkGetImageMemoryRequirements(device, im->image, &memRequirements);
  // VkDeviceMemory mem;
  AllocateMemmory(&memRequirements, &im->devicemem, 0);  // 0
  // AllocateMemmory(&memRequirements,mem);
  VLOG(5) << "vkBindImageMemory !!!!";

  vkBindImageMemory(device, im->image, im->devicemem, 0);
  CreateImageView(im->view, im->image, viewType, format);
  VLOG(5) << "vkCreateImage !!!!";
  images.push_back(im);
  return im;
}

VulkanBuf* VulkanDevice::CreateBuffer(uint32_t size) {
  VulkanBuf* pbuf = new VulkanBuf();
  VkBufferCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  info.pNext = nullptr;
  info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  info.size = size;
  info.pQueueFamilyIndices = &compute_queue_family_index;
  info.queueFamilyIndexCount = 1;
  info.flags = 0;

  VLOG(5) << "vkCreateBuffer !!!!";

  res = vkCreateBuffer(device, &info, nullptr, &pbuf->buf);
  if (res) LOG(FATAL) << "Vulkan Create Buffer error!";

  VkMemoryRequirements memRequirements;
  vkGetBufferMemoryRequirements(device, pbuf->buf, &memRequirements);
  AllocateMemmory(
      &memRequirements, &pbuf->devicemem, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
  VLOG(5) << "vkBindBufferMemory !!!!";

  vkBindBufferMemory(device, pbuf->buf, pbuf->devicemem, 0);
  pbuf->size = size;
  pbufs.push_back(pbuf);
  return pbuf;
}
void VulkanDevice::FlushBuffer(VulkanBuf* buf, uint64_t start, uint64_t size) {
  VkMappedMemoryRange range;
  range.memory = buf->devicemem;
  range.pNext = nullptr;
  range.offset = start;
  range.size = size;
  res = vkFlushMappedMemoryRanges(device, 1, &range);

  if (res) LOG(FATAL) << "Vulkan Flush Buffer error!";
}

void VulkanDevice::AllocateMemmory(VkMemoryRequirements* memoryRequirements,
                                   VkDeviceMemory* mem,
                                   VkFlags memTpye) {
  uint32_t memTypeIndex = 0;
  uint32_t mask = 1;
  auto typeBits = memoryRequirements->memoryTypeBits;
  for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
    if ((typeBits & 1) == 1) {
      if ((memory_properties.memoryTypes[i].propertyFlags & memTpye) ==
          memTpye) {
        memTypeIndex = i;
        break;
      }
    }
    typeBits >>= 1;
  }
  VkMemoryAllocateInfo info;
  info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  info.pNext = nullptr;
  info.allocationSize = memoryRequirements->size;
  info.memoryTypeIndex = memTypeIndex;
  VLOG(5) << "vkAllocateMemory !!!! memTypeIndex" << memTypeIndex;

  res = vkAllocateMemory(device, &info, nullptr, mem);
  if (res) LOG(FATAL) << "Vulkan AllocateMemory error!";
}
void VulkanDevice::CreateImageView(VkImageView& view,
                                   const VkImage& image,
                                   const VkImageViewType& type,
                                   const VkFormat& format) {
  VkImageViewCreateInfo info = {};
  info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  info.image = image;
  info.viewType = type;
  info.format = format;
  info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  info.subresourceRange.baseMipLevel = 0;
  info.subresourceRange.levelCount = 1;
  info.subresourceRange.baseArrayLayer = 0;
  info.subresourceRange.layerCount = 1;
  VLOG(5) << "vkCreateImageView !!!!";

  res = vkCreateImageView(device, &info, nullptr, &view);
  if (res) LOG(FATAL) << "Vulkan Create ImageView error!";
}

void VulkanDevice::MapMemory(VulkanBuf* bufdata,
                             const VkDeviceMemory memory,
                             const int start,
                             const int size) {
  if (size < 0) LOG(FATAL) << "Vulkan MapMemory size error, size < 0!";
  VLOG(5) << "vkMapMemory !!!!";

  res = vkMapMemory(device, memory, start, size, 0, &bufdata->mapped_ptr);
  if (res) LOG(FATAL) << "Vulkan MapMemory error!";
}

void VulkanDevice::UnMapMemory(const VulkanBuf* bufdata,
                               VkDeviceMemory memory) {
  VLOG(5) << "vkUnmapMemory !!!!";

  vkUnmapMemory(device, memory);
}

void VulkanDevice::CreateDescriptorSetLayout(
    const std::vector<VkDescriptorType>& types) {
  if (types.size() <= 0) {
    LOG(FATAL) << "Vulkan Create VkDescriptorType size is <= 0!";
  }

  std::map<VkDescriptorType, int> type_nums;

  std::vector<VkDescriptorSetLayoutBinding> bindings(types.size());
  for (int i = 0; i < types.size(); i++) {
    auto type = types[i];
    if (type_nums.find(type) == type_nums.end()) {
      type_nums[type] = 1;
    } else {
      type_nums[type] += 1;
    }
    bindings[i].binding = i;
    bindings[i].descriptorType = types[i];
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[i].pImmutableSamplers = nullptr;
  }

  for (auto& iter : type_nums) {
    VkDescriptorPoolSize pool;
    pool.descriptorCount = iter.second;
    pool.type = iter.first;
    pool_size.push_back(pool);
  }
  VkDescriptorSetLayoutCreateInfo info;
  info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  info.pNext = 0;
  info.flags = 0;
  info.bindingCount = types.size();
  info.pBindings = bindings.data();

  // if (vkdev->info.support_VK_KHR_push_descriptor)
  {
    // info.flags |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    // if set  instead pushed by vkCmdPushDescriptorSetKHR
  }

  res = vkCreateDescriptorSetLayout(device, &info, 0, &setLayout);
  if (res) LOG(FATAL) << "Vulkan Create DescriptorSetLayout error!";
}
void VulkanDevice::CreatePipelineLayout(
    const VkDescriptorSetLayout& setLayout) {
  VkPipelineLayoutCreateInfo layoutInfo = {};
  layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layoutInfo.setLayoutCount = 1;
  layoutInfo.pSetLayouts = &setLayout;

  res = vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout);
  if (res) LOG(FATAL) << "Vulkan Create DescriptorSetLayout error!";
}
VkShaderModule VulkanDevice::GetShaderModule(const std::string& name) {
  auto iter = shader_modules_map.find(name);
  if (iter == shader_modules_map.end()) {
    LOG(FATAL) << "The vulkan shader not support name:" << name;
  }
  return iter->second;
}
void VulkanDevice::CreateComputePipelines(
    const std::string& name, const std::vector<uint32_t>& localSize) {
  std::vector<VkSpecializationMapEntry> specializationMapEntry(
      localSize.size());
  std::shared_ptr<VkSpecializationInfo> specializationInfo =
      std::make_shared<VkSpecializationInfo>();

  if (localSize.size() > 0) {
    for (int i = 0; i < localSize.size(); i++) {
      specializationMapEntry[i].constantID = i + 1;
      specializationMapEntry[i].offset = sizeof(uint32_t) * i;
      specializationMapEntry[i].size = sizeof(uint32_t);
    }
    specializationInfo->pData = localSize.data();
    specializationInfo->dataSize =
        localSize.size() * sizeof(uint32_t); /*bytes*/
    specializationInfo->pMapEntries = specializationMapEntry.data();
    specializationInfo->mapEntryCount = specializationMapEntry.size();
  }
  VkPipelineShaderStageCreateInfo shaderInfo;
  ::memset(&shaderInfo, 0, sizeof(shaderInfo));
  shaderInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shaderInfo.pNext = 0;
  shaderInfo.flags = 0;
  shaderInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  shaderInfo.module = GetShaderModule(name);
  shaderInfo.pName = "main";
  shaderInfo.pSpecializationInfo = specializationInfo.get();

  VkComputePipelineCreateInfo info;
  ::memset(&info, 0, sizeof(info));
  info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  info.stage = shaderInfo;
  info.layout = pipelineLayout;
  info.pNext = 0;
  info.flags = 0;
  info.basePipelineHandle = 0;
  info.basePipelineIndex = 0;
  VLOG(5) << "vkCreateComputePipelines!";

  res = vkCreateComputePipelines(device, 0, 1, &info, 0, &pipeline);
  if (res) LOG(FATAL) << "Vulkan Create ComputePipelines error!" << res;
}

void VulkanDevice::CreateDescriptorPool(const uint32_t set_count) {
  // VkDescriptorPoolSize type_count[1] = {};
  // type_count[0].type = type;
  // type_count[0].descriptorCount = des_count;

  // int descriptor_set_count = 1;
  VkDescriptorPoolCreateInfo info[1] = {};
  info[0].sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  info[0].pNext = NULL;
  info[0].maxSets = set_count;  // descriptor_set_count
  info[0].poolSizeCount = pool_size.size();
  info[0].pPoolSizes = pool_size.data();
  res = vkCreateDescriptorPool(device, info, nullptr, descriptorpool);
  if (res) LOG(FATAL) << "Vulkan Create DescriptorPool error!" << res;
}

void VulkanDevice::AllocateDescriptorSets(
    const VkDescriptorPool& descPool, const VkDescriptorSetLayout& setLayout) {
  VkDescriptorSetAllocateInfo Info;
  ::memset(&Info, 0, sizeof(Info));
  Info.pNext = nullptr;
  Info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  Info.descriptorPool = descPool;
  Info.descriptorSetCount = 1;
  Info.pSetLayouts = &setLayout;

  res = vkAllocateDescriptorSets(device, &Info, descriptorSet);
  if (res) LOG(FATAL) << "Vulkan Create DescriptorSet error!" << res;
}

void VulkanDevice::CreateCommandPool() {
  VkCommandPoolCreateInfo Info = {};
  Info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  Info.pNext = NULL;
  Info.queueFamilyIndex = GetComputQueueIdx();
  Info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  LOG(INFO) << "vkCreateCommandPool";
  VkResult res = vkCreateCommandPool(GetVkDevice(), &Info, NULL, &cmdpool);
  if (res) LOG(FATAL) << "CommandPool Create error!";
}

void VulkanDevice::AllocateCommandBuffers() {
  VkCommandBufferAllocateInfo Info = {};
  Info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  Info.pNext = NULL;
  Info.commandPool = cmdpool;
  Info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  Info.commandBufferCount = 1;

  res = vkAllocateCommandBuffers(GetVkDevice(), &Info, &cmd);
  if (res) LOG(FATAL) << "Vulkan Allocate CommandPool Create error!";
}

void VulkanDevice::AllocateCommandBuffers(VkCommandBuffer* cmd) {
  VkCommandBufferAllocateInfo Info = {};
  Info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  Info.pNext = NULL;
  Info.commandPool = cmdpool;
  Info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  Info.commandBufferCount = 1;

  res = vkAllocateCommandBuffers(GetVkDevice(), &Info, cmd);
  if (res) LOG(FATAL) << "Vulkan Allocate CommandPool Create error!";
}

void VulkanDevice::BeginCommandBuffer(VkCommandBuffer cmd,
                                      VkCommandBufferUsageFlags flag) {
  VkCommandBufferBeginInfo Info;

  Info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
  Info.pNext = nullptr, Info.flags = flag, Info.pInheritanceInfo = nullptr,
  vkResetCommandBuffer(cmd, 0);
  res = vkBeginCommandBuffer(cmd, &Info);
  if (res) LOG(FATAL) << "Vulkan Command Begin error!";
}

void VulkanDevice::EndCommandBuffer(VkCommandBuffer cmd) {
  res = vkEndCommandBuffer(cmd);

  if (res) LOG(FATAL) << "Vulkan Command Begin error!";
}

void VulkanDevice::UpdateDescriptorSets(
    VkDevice device, int size, VkBuffer buf, int index, VkDescriptorType type) {
  int bufoffset = 0;
  VkWriteDescriptorSet writeSet;
  ::memset(&writeSet, 0, sizeof(writeSet));
  VkDescriptorBufferInfo sourceInfo;
  sourceInfo.buffer = buf;
  sourceInfo.offset = bufoffset;
  sourceInfo.range = size;
  writeSet.descriptorCount = 1;
  writeSet.descriptorType = type;
  writeSet.dstBinding = index;
  writeSet.pBufferInfo = &sourceInfo;
  writeSet.dstSet = *descriptorSet;

  vkUpdateDescriptorSets(device, 1, &writeSet, 0, nullptr);
}

void VulkanDevice::UpdateDescriptorSets(VkDevice device,
                                        const VulkanMem* vulkan_mem,
                                        int index,
                                        VkDescriptorType type,
                                        VkImageLayout layout) {
  VkWriteDescriptorSet writeSet;
  ::memset(&writeSet, 0, sizeof(writeSet));
  VkDescriptorBufferInfo bufinfo;
  VkDescriptorImageInfo imginfo;
  if (vulkan::MemType::BUF == vulkan_mem->type) {
    vulkan::VulkanBuf* vulkan_buf = (vulkan::VulkanBuf*)vulkan_mem->mem.buf;
    int bufoffset = 0;
    bufinfo.buffer = vulkan_buf->buf;
    bufinfo.offset = bufoffset;
    bufinfo.range = vulkan_buf->size;

    writeSet.pBufferInfo = &bufinfo;
  } else {
    vulkan::VulkanImage* vulkan_image =
        (vulkan::VulkanImage*)vulkan_mem->mem.image;
    imginfo.imageView = vulkan_image->view;
    imginfo.imageLayout = layout;  /// VK_IMAGE_LAYOUT_GENERAL
    imginfo.sampler = sampler;

    writeSet.pImageInfo = &imginfo;
  }
  writeSet.descriptorCount = 1;
  writeSet.descriptorType = type;
  writeSet.dstBinding = index;
  writeSet.dstSet = *descriptorSet;
  vkUpdateDescriptorSets(device, 1, &writeSet, 0, nullptr);
}

void VulkanDevice::QueueSubmit(VkQueue compute_queue,
                               VkCommandBuffer cmd,
                               VkFence fence) {
  VkSubmitInfo Info;

  Info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  Info.pNext = nullptr;
  Info.waitSemaphoreCount = 0;
  Info.pWaitSemaphores = nullptr;
  Info.pWaitDstStageMask = nullptr;
  Info.commandBufferCount = 1;
  Info.pCommandBuffers = &cmd;
  Info.signalSemaphoreCount = 0;
  Info.pSignalSemaphores = nullptr;
  VkResult res = vkQueueSubmit(compute_queue, 1, &Info, fence);
  if (res) LOG(FATAL) << "vkQueueSubmit error";
}

void VulkanDevice::QueueSubmit(VkQueue compute_queue,
                               std::vector<VkCommandBuffer>* cmds,
                               VkFence fence) {
  VkSubmitInfo Info;

  Info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  Info.pNext = nullptr;
  Info.waitSemaphoreCount = 0;
  Info.pWaitSemaphores = nullptr;
  Info.pWaitDstStageMask = nullptr;
  Info.commandBufferCount = (uint32_t)cmds->size();
  Info.pCommandBuffers = cmds->data();
  Info.signalSemaphoreCount = 0;
  Info.pSignalSemaphores = nullptr;
  VkResult res = vkQueueSubmit(compute_queue, 1, &Info, fence);
  if (res) LOG(FATAL) << "vkQueueSubmit error, error code: " << res;
}

VulkanDevice::~VulkanDevice() {
  if (VK_NULL_HANDLE != inst) {
    vkDestroyInstance(inst, nullptr);
    inst = VK_NULL_HANDLE;
  }
}

#endif  // LITE_WITH_VULKAN
}  // namespace vulkan
}  // namespace lite
}  // namespace paddle
