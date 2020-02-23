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

#include "lite/kernels/vulkan/activation_compute.h"
#include "lite/backends/vulkan/vk_buffer.h"
#include "lite/backends/vulkan/vk_device.h"
namespace paddle {
namespace lite {
namespace kernels {
namespace vulkan {

void ReluCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<VULKANContext>();
  auto input = param.X;
  auto output = param.Out;
  lite::DDim input_dim = input->dims();
  lite::DDim output_dim = output->dims();
  std::vector<int> input_dim_v;
  for (int i = 0; i < input_dim.size(); i++) {
    input_dim_v.push_back(input_dim.data()[i]);
    LOG(INFO) << input_dim_v[i];
  }

  auto device = ctx.device();
  lite::vulkan::VulkanMem* imagedataout =
      (lite::vulkan::VulkanMem*)output->mutable_data<float>(
          TARGET(kVULKAN), input_dim.data()[3], input_dim.data()[2]);
  lite::vulkan::VulkanMem* buf_param =
      (lite::vulkan::VulkanMem*)TargetWrapper<TARGET(kVULKAN)>::Malloc(
          input_dim_v.size() * sizeof(int));
  CopySync<TARGET(kVULKAN)>(buf_param,
                            input_dim_v.data(),
                            input_dim_v.size() * sizeof(int),
                            IoDirection::HtoD);

  std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
  device->CreateDescriptorSetLayout(types);

  device->CreatePipelineLayout(device->setLayout);
  LOG(INFO) << "device->shader_modules" << device->shader_modules.size();

  device->CreateComputePipelines("relu");

  device->CreateDescriptorPool(1);
  device->AllocateDescriptorSets(*(device->descriptorpool), device->setLayout);
  // int bufsize = data_size * sizeof(float);
  int input_index = 0;
  const lite::vulkan::VulkanMem* bufdata =
      input->data<lite::vulkan::VulkanMem>();
  device->UpdateDescriptorSets(device->GetVkDevice(),
                               bufdata,
                               input_index,
                               types[input_index],
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
  int output_index = 1;
  device->UpdateDescriptorSets(device->GetVkDevice(),
                               imagedataout,
                               output_index,
                               types[output_index],
                               VK_IMAGE_LAYOUT_GENERAL);

  int param_index = 2;
  device->UpdateDescriptorSets(device->GetVkDevice(),
                               buf_param,
                               param_index,
                               types[param_index],
                               VK_IMAGE_LAYOUT_GENERAL);

  device->AllocateCommandBuffers();
  device->BeginCommandBuffer(device->cmd, 0);
  // LOG(INFO)<<"vkCmdBindPipeline !!!!\n";
  vkCmdBindPipeline(
      device->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, device->pipeline);

  // Bind descriptor set.
  vkCmdBindDescriptorSets(device->cmd,
                          VK_PIPELINE_BIND_POINT_COMPUTE,
                          device->pipelineLayout,
                          0,
                          1,
                          device->descriptorSet,
                          0,
                          nullptr);
  // VkDescriptorPool descriptor_pool[1] = {};
  // res = vkCreateDescriptorPool(vk_dev->device, poolinfo, NULL,
  // descriptor_pool);
  LOG(INFO) << "vkCmdBindDescriptorSets!!!!";

  vkCmdDispatch(device->cmd,
                (input_dim_v[3] + 15) / 16,
                (input_dim_v[2] + 15) / 16,
                (input_dim_v[1] + 3) / 4);
  LOG(INFO) << "kernel relu x:" << (input_dim_v[3] + 15) / 16;
  LOG(INFO) << "kernel relu x:" << (input_dim_v[2] + 15) / 16;
  LOG(INFO) << "kernel relu x:" << (input_dim_v[1] + 3) / 4;
  LOG(INFO) << "kernel device ptr:" << device;
  LOG(INFO) << "kernel cmd ptr:" << device->cmd;
  device->EndCommandBuffer(device->cmd);

  vkResetFences(device->GetVkDevice(), 1, &device->fence);
  device->QueueSubmit(device->compute_queue, device->cmd, device->fence);

  VkResult res = vkWaitForFences(
      device->GetVkDevice(), 1, &device->fence, VK_TRUE, UINT64_MAX);
  if (res) LOG(FATAL) << "vkWaitForFences error";

  // device->EndCommandBuffer(device->cmd);

  // vkResetFences(device->GetVkDevice(), 1, &device->fence);
  // device->QueueSubmit(device->compute_queue,device->cmd,device->fence);

  // VkResult res = vkWaitForFences(
  //     device->GetVkDevice(), 1, &device->fence, VK_TRUE, UINT64_MAX);
  // if (res) LOG(FATAL) << "vkWaitForFences error";

  // LOG(INFO) << "vkWaitForFences ";

  // for (int i = 0; i < 128; i++) {
  //   // (reinterpret_cast<float*>(bufdata->mapped_ptr))[i] = i*1.f;
  //   LOG(INFO) << (reinterpret_cast<float*>(bufdata->mapped_ptr))[i]
  //             << "   |  out:"
  //             << (reinterpret_cast<float*>(bufdataout->mapped_ptr))[i];
  // }
}
void ReluCompute::PrepareForRun() {
  // auto& param = this->Param<param_t>();
  // auto& ctx = this->ctx_->template As<ARMContext>();
  // auto x_dims = param.X->dims();
  // auto x_data = param.X->data<float>();
  // auto output_data = param.Out->mutable_data<float>();
  //   lite::arm::math::act_relu<float>(
  //       x_data, output_data, x_dims.production(), ctx.threads());
}

}  // namespace vulkan
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(relu,
                     kVULKAN,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::vulkan::ReluCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kVULKAN))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kVULKAN))})
    .Finalize();
