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

#include "lite/backends/vulkan/target_wrapper.h"
#include <string>
#include "lite/backends/vulkan/vk_device.h"
#include "lite/core/context.h"
#include "lite/utils/all.h"

namespace paddle {
namespace lite {

void* TargetWrapperVulkan::Malloc(size_t size) {
  vulkan::VulkanMem* vulkan_mem = new vulkan::VulkanMem();
  auto ctx_ = ContextScheduler::Global().NewContext(TARGET(kVULKAN));
  auto& ctx = ctx_->template As<VULKANContext>();

  auto device = ctx.device();
  LOG(INFO) << "malocsize vulkan:" << size;
  vulkan_mem->mem.buf = (vulkan::VulkanBuf*)device->CreateBuffer(size);
  vulkan_mem->type = vulkan::MemType::BUF;
  // size should be divisable by 16??
  return vulkan_mem;
}

template <>
void* TargetWrapperVulkan::MallocImage<float>(uint32_t img_w, uint32_t img_h) {
  vulkan::VulkanMem* vulkan_mem = new vulkan::VulkanMem();

  auto ctx_ = ContextScheduler::Global().NewContext(TARGET(kVULKAN));
  auto& ctx = ctx_->template As<VULKANContext>();

  auto device = ctx.device();

  // size should be divisable by 16??

  uint32_t depth = 4;
  vulkan_mem->mem.image =
      (vulkan::VulkanImage*)device->CreateImage(VK_IMAGE_TYPE_3D,
                                                VK_IMAGE_VIEW_TYPE_3D,
                                                depth,
                                                img_w,
                                                img_h,
                                                VK_FORMAT_R16G16B16A16_SFLOAT);
  vulkan_mem->type = vulkan::MemType::IMAGE;

  return vulkan_mem;
}

void TargetWrapperVulkan::Free(void* ptr) {}

void TargetWrapperVulkan::MemcpySync(void* dst,
                                     const void* src,
                                     size_t size,
                                     IoDirection dir) {
  auto ctx_ = ContextScheduler::Global().NewContext(TARGET(kVULKAN));
  auto& ctx = ctx_->template As<VULKANContext>();
  vulkan::VulkanMem* vulkan_mem;
  vulkan::VulkanBuf* vulkan_buf;
  switch (dir) {
    case IoDirection::HtoD:
      vulkan_mem = (vulkan::VulkanMem*)dst;
      if (vulkan::MemType::BUF == vulkan_mem->type) {
        vulkan_buf = (vulkan::VulkanBuf*)vulkan_mem->mem.buf;
        ctx.device()->MapMemory(
            vulkan_buf, vulkan_buf->devicemem, 0, vulkan_buf->size);
        memcpy(vulkan_buf->mapped_ptr, src, size);

      } else {
        LOG(FATAL) << "Mem Type is  not support";
      }

      break;
    case IoDirection::DtoH:
      vulkan_mem = (vulkan::VulkanMem*)src;
      if (vulkan::MemType::BUF == vulkan_mem->type) {
        vulkan_buf = (vulkan::VulkanBuf*)vulkan_mem->mem.buf;

        ctx.device()->MapMemory(
            vulkan_buf, vulkan_buf->devicemem, 0, vulkan_buf->size);

        memcpy(dst, vulkan_buf->mapped_ptr, size);
      } else {
        LOG(FATAL) << "Mem Type is  not support";
      }
      break;
    default:
      LOG(FATAL) << "Unsupported IoDirection " << static_cast<int>(dir);
      break;
  }
}
void TargetWrapperVulkan::MemcpyAsync(void* dst,
                                      const void* src,
                                      size_t size,
                                      IoDirection dir,
                                      const stream_t& stream) {}

void TargetWrapperVulkan::MemsetSync(void* devPtr, int value, size_t count) {}

void TargetWrapperVulkan::MemsetAsync(void* devPtr,
                                      int value,
                                      size_t count,
                                      const stream_t& stream) {}
}  // namespace lite
}  // namespace paddle
