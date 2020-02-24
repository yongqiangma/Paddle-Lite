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

#include "lite/kernels/vulkan/vulkan_utils.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace vulkan {

DDim InitImageDimInfoWith(const DDim& tensor_dim) {
  size_t new_dims[] = {1, 1, 1, 1};
  for (size_t j = 0; j < tensor_dim.size(); ++j) {
    new_dims[4 - tensor_dim.size() + j] = tensor_dim[j];
  }
  size_t N, C, H, W;
  N = new_dims[0];
  C = new_dims[1];
  H = new_dims[2];
  W = new_dims[3];
  size_t width = W * ((C + 3) / 4);
  size_t height = H * N;
  return DDim(
      std::vector<DDim::value_type>({static_cast<DDim::value_type>(width),
                                     static_cast<DDim::value_type>(height)}));
}

DDim H4WInitImageDimInfoWith(const DDim& tensor_dim) {
  if (tensor_dim.size() <= 2) {
    size_t tdim[2] = {1, 1};
    if (tensor_dim.size() == 1) {
      tdim[1] = tensor_dim[0];
    } else {
      tdim[0] = tensor_dim[0];
      tdim[1] = tensor_dim[1];
    }
    size_t width = tdim[1];
    size_t height = (tdim[0] + 3) / 4;
    return DDim(
        std::vector<DDim::value_type>({static_cast<DDim::value_type>(width),
                                       static_cast<DDim::value_type>(height)}));

  } else {
    size_t new_dims[] = {1, 1, 1, 1};
    for (size_t j = 0; j < tensor_dim.size(); ++j) {
      new_dims[4 - tensor_dim.size() + j] = tensor_dim[j];
    }
    size_t N, C, H, W;
    N = new_dims[0];
    C = new_dims[1];
    H = new_dims[2];
    W = new_dims[3];
    size_t width = C;
    size_t height = (N + 3) / 4;

    // width_of_one_block_ = W;
    // height_of_one_block_ = H;
    // c_block_ = width / W;

    return DDim(
        std::vector<DDim::value_type>({static_cast<DDim::value_type>(width),
                                       static_cast<DDim::value_type>(height)}));
  }
}

DDim HW4InitImageDimInfoWith(const DDim& tensor_dim) {
  if (tensor_dim.size() <= 2) {
    size_t tdim[2] = {1, 1};
    if (tensor_dim.size() == 1) {
      tdim[1] = tensor_dim[0];
    } else {
      tdim[0] = tensor_dim[0];
      tdim[1] = tensor_dim[1];
    }
    size_t width = (tdim[1] + 3) / 4;
    size_t height = tdim[0];

    // width_of_one_block_ = width;
    // height_of_one_block_ = height;
    // c_block_ = 1;

    return DDim(
        std::vector<DDim::value_type>({static_cast<DDim::value_type>(width),
                                       static_cast<DDim::value_type>(height)}));

  } else {
    LOG(FATAL) << "input dim >2";
  }
}

DDim ComputeOutputDims(DDim image_dims, BufTYPE type) {
  DDim vulkan_image_dim;
  switch (type) {
    case BufTYPE::NCHW:
      vulkan_image_dim = InitImageDimInfoWith(image_dims);
      break;
    case BufTYPE::HW4:
      vulkan_image_dim = HW4InitImageDimInfoWith(image_dims);
      break;
    case BufTYPE::H4W:
      vulkan_image_dim = H4WInitImageDimInfoWith(image_dims);
      break;
    default:
      vulkan_image_dim = InitImageDimInfoWith(image_dims);
      break;
  }
  return vulkan_image_dim;
}

std::string Get2ImageShaderName(BufTYPE type) {
  std::string shader_name;
  switch (type) {
    case BufTYPE::NCHW:
      shader_name = "nchw2image";
      break;
    case BufTYPE::HW4:
      shader_name = "hw2image";
      break;
    case BufTYPE::H4W:
      shader_name = "wh2image";
      break;
    default:
      shader_name = "nchw2image";
      break;
  }
  return shader_name;
}

std::string Get2BufShaderName(BufTYPE type) {
  std::string shader_name;
  switch (type) {
    case BufTYPE::NCHW:
      shader_name = "image2nchw";
      break;
    case BufTYPE::HW4:
      shader_name = "image2hw4";
      break;
    case BufTYPE::H4W:
      shader_name = "image2h4w";
      break;
    default:
      shader_name = "image2nchw";
      break;
  }
  return shader_name;
}
void CopyToVulkan(std::shared_ptr<lite::vulkan::VulkanDevice> device,
                  const lite::Tensor* host,
                  lite::Tensor* vk) {
  lite::DDim h_dims = host->dims();
  vk->Resize(h_dims);
  auto* dst = vk->mutable_data<float>(TARGET(kVULKAN));
  // auto in_data = host->data<float>();
  //   for(int i = 0; i< 100; i++){
  //   LOG(INFO)<<"io_copy in_data "<<h_dims<<"  "<<i<<"  :"<<in_data[i];
  // }
  CopySync<TARGET(kVULKAN)>(dst,
                            host->data<float>(),
                            h_dims.production() * sizeof(float),
                            IoDirection::HtoD);
}

void CopyToHost(std::shared_ptr<lite::vulkan::VulkanDevice> device,
                const lite::Tensor* vk,
                lite::Tensor* host) {
  lite::DDim vk_dims = vk->dims();
  host->Resize(vk_dims);
  LOG(INFO) << "CopyToHost";
  auto* dst = host->mutable_data<float>();
  LOG(INFO) << "CopyToHost";
  CopySync<TARGET(kVULKAN)>(dst,
                            vk->data<float>(),
                            sizeof(float) * vk_dims.production(),
                            IoDirection::DtoH);
  LOG(INFO) << "CopyToHost";
}

void VulkanMatrixTrans(std::shared_ptr<lite::vulkan::VulkanDevice> device,
                       const lite::Tensor* input,
                       lite::Tensor* output) {
  lite::DDim input_dim = input->dims();
  if (!(input_dim.size() == 2))
    LOG(FATAL) << "VulkanMatrixTrans input dim.size must = 2 ";
  std::vector<int> input_dim_v;
  for (int i = 0; i < input_dim.size(); i++) {
    input_dim_v.push_back(input_dim[i]);
  }

  output->Resize(input_dim);
  lite::vulkan::VulkanMem* out_mem =
      (lite::vulkan::VulkanMem*)output->mutable_data<float>(TARGET(kVULKAN));
  lite::vulkan::VulkanMem* buf_param =
      (lite::vulkan::VulkanMem*)TargetWrapper<TARGET(kVULKAN)>::Malloc(
          input_dim_v.size() * sizeof(int));

  CopySync<TARGET(kVULKAN)>(buf_param,
                            input_dim_v.data(),
                            input_dim_v.size() * sizeof(int),
                            IoDirection::HtoD);

  std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
  device->CreateDescriptorSetLayout(types);

  device->CreatePipelineLayout(device->setLayout);

  device->CreateComputePipelines("matrixtrans");

  device->CreateDescriptorPool(1);
  device->AllocateDescriptorSets(*(device->descriptorpool), device->setLayout);

  int input_index = 0;
  const lite::vulkan::VulkanMem* in_mem =
      input->data<lite::vulkan::VulkanMem>();

  device->UpdateDescriptorSets(device->GetVkDevice(),
                               in_mem,
                               input_index,
                               types[input_index],
                               VK_IMAGE_LAYOUT_GENERAL);
  int output_index = 1;
  device->UpdateDescriptorSets(device->GetVkDevice(),
                               out_mem,
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

  vkCmdDispatch(device->cmd, 1, (input_dim[1] + 15) / 16, 1);
  device->EndCommandBuffer(device->cmd);

  vkResetFences(device->GetVkDevice(), 1, &device->fence);
  device->QueueSubmit(device->compute_queue, device->cmd, device->fence);

  VkResult res = vkWaitForFences(
      device->GetVkDevice(), 1, &device->fence, VK_TRUE, UINT64_MAX);
  if (res) LOG(FATAL) << "vkWaitForFences error";
}

void VulkanBuf2Image(std::shared_ptr<lite::vulkan::VulkanDevice> device,
                     const lite::Tensor* input,
                     lite::Tensor* output,
                     BufTYPE type) {
  lite::DDim input_dim = input->dims();
  std::vector<int> input_dim_v;
  if (input_dim.size() == 1) input_dim_v.push_back(1);
  for (int i = 0; i < input_dim.size(); i++) {
    input_dim_v.push_back(input_dim[i]);
  }
  DDim out_image_dim = ComputeOutputDims(input_dim, type);
  std::string shader_name = Get2ImageShaderName(type);

  for (int i = 0; i < out_image_dim.size(); i++) {
    input_dim_v.push_back(out_image_dim[i]);
    LOG(INFO) << "image_dim W: " << out_image_dim[0]
              << "  H: " << out_image_dim[1];
  }

  output->Resize(input_dim);
  lite::vulkan::VulkanMem* imagedataout =
      (lite::vulkan::VulkanMem*)output->mutable_data<float>(
          TARGET(kVULKAN), out_image_dim[0], out_image_dim[1]);
  lite::vulkan::VulkanMem* buf_param =
      (lite::vulkan::VulkanMem*)TargetWrapper<TARGET(kVULKAN)>::Malloc(
          input_dim_v.size() * sizeof(int));

  CopySync<TARGET(kVULKAN)>(buf_param,
                            input_dim_v.data(),
                            input_dim_v.size() * sizeof(int),
                            IoDirection::HtoD);

  std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
  device->CreateDescriptorSetLayout(types);

  device->CreatePipelineLayout(device->setLayout);

  device->CreateComputePipelines(shader_name.c_str());

  device->CreateDescriptorPool(1);
  device->AllocateDescriptorSets(*(device->descriptorpool), device->setLayout);

  int input_index = 0;
  const lite::vulkan::VulkanMem* bufdata =
      input->data<lite::vulkan::VulkanMem>();

  device->UpdateDescriptorSets(device->GetVkDevice(),
                               bufdata,
                               input_index,
                               types[input_index],
                               VK_IMAGE_LAYOUT_GENERAL);
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

  vkCmdDispatch(device->cmd,
                (out_image_dim[0] + 15) / 16,
                (out_image_dim[1] + 15) / 16,
                1);
  device->EndCommandBuffer(device->cmd);

  vkResetFences(device->GetVkDevice(), 1, &device->fence);
  device->QueueSubmit(device->compute_queue, device->cmd, device->fence);

  VkResult res = vkWaitForFences(
      device->GetVkDevice(), 1, &device->fence, VK_TRUE, UINT64_MAX);
  if (res) LOG(FATAL) << "vkWaitForFences error";
}

void VulkanPrintImage(std::shared_ptr<lite::vulkan::VulkanDevice> device,
                      const lite::Tensor* image,
                      BufTYPE type) {
  auto image_dims = image->dims();
  std::vector<int> image_dim_v;
  for (int i = 0; i < 4 - image_dims.size(); i++) {
    image_dim_v.push_back(1);
  }

  for (int i = 0; i < image_dims.size(); i++) {
    image_dim_v.push_back(image_dims[i]);
    // LOG(INFO) << "image_dim_v " << image_dims[i];
  }

  DDim vulkan_image_dim = ComputeOutputDims(image_dims, type);

  // DDim vulkan_image_dim = InitImageDimInfoWith(image_dims);
  for (int i = 0; i < vulkan_image_dim.size(); i++) {
    image_dim_v.push_back(vulkan_image_dim[i]);
  }
  // LOG(INFO) << "filter w :"<<vulkan_image_dim[0] <<"
  // h:"<<vulkan_image_dim[1];
  lite::Tensor output;
  std::vector<int64_t> output_dim{
      1, 4, vulkan_image_dim[1], vulkan_image_dim[0]};
  output.Resize(output_dim);
  lite::vulkan::VulkanMem* bufdataout =
      (lite::vulkan::VulkanMem*)output.mutable_data<float>(TARGET(kVULKAN));
  lite::vulkan::VulkanMem* buf_param =
      (lite::vulkan::VulkanMem*)TargetWrapper<TARGET(kVULKAN)>::Malloc(
          image_dim_v.size() * sizeof(int));

  CopySync<TARGET(kVULKAN)>(buf_param,
                            image_dim_v.data(),
                            image_dim_v.size() * sizeof(int),
                            IoDirection::HtoD);

  std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
  device->CreateDescriptorSetLayout(types);

  device->CreatePipelineLayout(device->setLayout);

  device->CreateComputePipelines("printimage");

  device->CreateDescriptorPool(1);
  device->AllocateDescriptorSets(*(device->descriptorpool), device->setLayout);
  int input_index = 0;
  const lite::vulkan::VulkanMem* image_mem =
      image->data<lite::vulkan::VulkanMem>();

  device->UpdateDescriptorSets(device->GetVkDevice(),
                               image_mem,
                               input_index,
                               types[input_index],
                               VK_IMAGE_LAYOUT_GENERAL);
  int output_index = 1;
  device->UpdateDescriptorSets(device->GetVkDevice(),
                               bufdataout,
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

  vkCmdDispatch(device->cmd,
                (vulkan_image_dim[0] + 15) / 16,
                (vulkan_image_dim[1] + 15) / 16,
                1);
  device->EndCommandBuffer(device->cmd);

  vkResetFences(device->GetVkDevice(), 1, &device->fence);
  device->QueueSubmit(device->compute_queue, device->cmd, device->fence);

  VkResult res = vkWaitForFences(
      device->GetVkDevice(), 1, &device->fence, VK_TRUE, UINT64_MAX);
  if (res) LOG(FATAL) << "vkWaitForFences error";

  float* out_cpu_data2 =
      static_cast<float*>(malloc(sizeof(float) * output.numel()));
  CopySync<TARGET(kVULKAN)>(reinterpret_cast<void*>(out_cpu_data2),
                            reinterpret_cast<void*>(bufdataout),
                            sizeof(float) * output.numel(),
                            IoDirection::DtoH);
  LOG(INFO) << "PrintImage:" << output.numel() << ": " << output.dims();
  for (int i = 0; i < output.numel(); i++) {
    LOG(INFO) << "index:" << i << ": "
              << (reinterpret_cast<float*>(out_cpu_data2))[i];
  }
}

// void CopyToHost(std::shared_ptr<lite::vulkan::VulkanDevice> device, const
// lite::Tensor* vk, lite::Tensor* host){
//   output->Resize(input_dim);

//   auto*  out_cpu = output->mutable_data<float>();
//   CopySync<TARGET(kVULKAN)>((void*)out_cpu,
//                             (void*)bufdataout,
//                             sizeof(float) * vulkan_output.numel(),
//                             IoDirection::DtoH);
// }

void VulkanImage2Buf(std::shared_ptr<lite::vulkan::VulkanDevice> device,
                     const lite::Tensor* image,
                     lite::Tensor* output,
                     BufTYPE type) {
  //   auto& param = this->Param<param_t>();
  // auto& ctx = this->ctx_->template As<VULKANContext>();
  // auto input = param.X;
  // auto output = param.Out;
  lite::DDim input_dim = image->dims();
  std::vector<int> input_dim_v;
  for (int i = 0; i < input_dim.size(); i++) {
    input_dim_v.push_back(input_dim.data()[i]);
  }

  // auto device = ctx.device();
  DDim in_image_dim = ComputeOutputDims(input_dim, type);
  // LOG(INFO)<<"VulkanImage2Buf  in_image_dim:"<<in_image_dim;

  std::string shader_name = Get2BufShaderName(type);

  for (int i = 0; i < in_image_dim.size(); i++) {
    input_dim_v.push_back(in_image_dim[i]);
  }
  // lite::Tensor output;
  output->Resize(input_dim);
  // lite::Tensor vulkan_output;
  // vulkan_output.Resize(input_dim);
  lite::vulkan::VulkanMem* bufdataout =
      (lite::vulkan::VulkanMem*)output->mutable_data<float>(TARGET(kVULKAN));
  lite::vulkan::VulkanMem* buf_param =
      (lite::vulkan::VulkanMem*)TargetWrapper<TARGET(kVULKAN)>::Malloc(
          input_dim_v.size() * sizeof(int));

  CopySync<TARGET(kVULKAN)>(buf_param,
                            input_dim_v.data(),
                            input_dim_v.size() * sizeof(int),
                            IoDirection::HtoD);

  std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
  device->CreateDescriptorSetLayout(types);

  device->CreatePipelineLayout(device->setLayout);

  device->CreateComputePipelines(shader_name);

  device->CreateDescriptorPool(1);
  device->AllocateDescriptorSets(*(device->descriptorpool), device->setLayout);
  int input_index = 0;
  const lite::vulkan::VulkanMem* bufdata =
      image->data<lite::vulkan::VulkanMem>();

  device->UpdateDescriptorSets(device->GetVkDevice(),
                               bufdata,
                               input_index,
                               types[input_index],
                               VK_IMAGE_LAYOUT_GENERAL);
  int output_index = 1;
  device->UpdateDescriptorSets(device->GetVkDevice(),
                               bufdataout,
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

  VLOG(5) << "vkCmdBindPipeline";
  vkCmdBindPipeline(
      device->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, device->pipeline);

  // Bind descriptor set.
  VLOG(5) << "vkCmdBindDescriptorSets";
  vkCmdBindDescriptorSets(device->cmd,
                          VK_PIPELINE_BIND_POINT_COMPUTE,
                          device->pipelineLayout,
                          0,
                          1,
                          device->descriptorSet,
                          0,
                          nullptr);

  vkCmdDispatch(
      device->cmd, (in_image_dim[0] + 15) / 16, (in_image_dim[1] + 15) / 16, 1);
  device->EndCommandBuffer(device->cmd);

  /*
    vkResetFences(device->GetVkDevice(), 1, &device->fence);
    device->QueueSubmit(device->compute_queue, device->cmd, device->fence);

    VkResult res = vkWaitForFences(
        device->GetVkDevice(), 1, &device->fence, VK_TRUE, UINT64_MAX);
    if (res) LOG(FATAL) << "vkWaitForFences error";
  */
  /*
    auto* out_cpu = output->mutable_data<float>();
    CopySync<TARGET(kVULKAN)>((void*)out_cpu,
                              (void*)bufdataout,
                              sizeof(float) * vulkan_output.numel(),
                              IoDirection::DtoH);
  */
  // for(int i = 0; i<100; i++){
  //   LOG(INFO)<<i<<"  :"<<out_cpu[i];

  // }
}

void VulkanRun(std::shared_ptr<lite::vulkan::VulkanDevice> device) {
  vkResetFences(device->GetVkDevice(), 1, &device->fence);
  device->QueueSubmit(device->compute_queue, &device->cmds, device->fence);
  // LOG(INFO)<<"device_cmds_size:"<<device->cmds.size();
  device->cmds.clear();
  VkResult res = vkWaitForFences(
      device->GetVkDevice(), 1, &device->fence, VK_TRUE, UINT64_MAX);
  if (res) LOG(FATAL) << "vkWaitForFences error";
}
void PrintTensor(const lite::Tensor* in, const std::string name) {
  auto ptr = in->data<float>();
  float sum = 0;
  for (int i = 0; i < in->numel(); ++i) {
    sum += ptr[i];
    LOG(INFO) << name << ": " << i << " :" << ptr[i];
  }
  float output_agv = sum / in->numel();
  LOG(INFO) << "------------------" << name << in->dims()
            << "-------------:" << output_agv;
}
}  // namespace vulkan
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
