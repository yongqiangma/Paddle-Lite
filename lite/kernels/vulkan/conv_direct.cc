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

#include "lite/kernels/vulkan/conv_direct.h"
#include <string>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace vulkan {

typedef struct {
  int input[4];
  int filter[4];
  int out[4];
  int strides[2];
  int pads[2];
  int dilations[2];
  int out_im[2];
  bool flag[2];
} conv_shader_param;

// void DirectConv::Run() {
void DirectConv::PrepareForRun() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<VULKANContext>();
  auto input = param.x;
  auto bias = param.bias;
  auto fuse_relu = param.fuse_relu;
  auto filter = param.filter;
  auto output = param.output;
  auto stride = param.strides;
  auto pad = param.paddings;
  auto dilation = param.dilations;
  LOG(INFO) << "vulkan_conv_d";
  const auto b_data = param.bias ? param.bias->data<float>() : nullptr;

  lite::DDim input_dim = input->dims();
  lite::DDim output_dim = output->dims();
  lite::DDim filter_dim = filter->dims();
  std::vector<int> input_dim_v;

  for (int i = 0; i < input_dim.size(); i++) {
    input_dim_v.push_back(input_dim.data()[i]);
    // LOG(INFO) << input_dim_v[i];
  }

  LOG(INFO) << "vulkan_conv_d";
  conv_shader_param shader_param;
  shader_param.input[0] = input_dim_v.data()[0];
  shader_param.input[1] = input_dim_v.data()[1];
  shader_param.input[2] = input_dim_v.data()[2];
  shader_param.input[3] = input_dim_v.data()[3];

  shader_param.out[0] = output_dim.data()[0];
  shader_param.out[1] = output_dim.data()[1];
  shader_param.out[2] = output_dim.data()[2];
  shader_param.out[3] = output_dim.data()[3];

  shader_param.filter[0] = filter_dim.data()[0];
  shader_param.filter[1] = filter_dim.data()[1];
  shader_param.filter[2] = filter_dim.data()[2];
  shader_param.filter[3] = filter_dim.data()[3];
  DDim out_image_dim = InitImageDimInfoWith(output_dim);
  shader_param.out_im[0] = out_image_dim[0];
  shader_param.out_im[1] = out_image_dim[1];

  shader_param.strides[0] = stride[0];
  shader_param.strides[1] = stride[1];

  shader_param.dilations[0] = dilation->data()[0];
  shader_param.dilations[1] = dilation->data()[1];

  shader_param.pads[0] = pad->data()[0];
  shader_param.pads[1] = pad->data()[1];

  shader_param.flag[0] = bias;
  shader_param.flag[1] = fuse_relu;

  device = ctx.device();
  lite::vulkan::VulkanMem* imagedataout =
      (lite::vulkan::VulkanMem*)output->mutable_data<float>(
          TARGET(kVULKAN), out_image_dim[0], out_image_dim[1]);

  lite::vulkan::VulkanMem* buf_param =
      (lite::vulkan::VulkanMem*)TargetWrapper<TARGET(kVULKAN)>::Malloc(
          sizeof(conv_shader_param));
  CopySync<TARGET(kVULKAN)>(
      buf_param, &shader_param, sizeof(conv_shader_param), IoDirection::HtoD);
  lite::Tensor filter_im;

  if (device == nullptr) LOG(FATAL) << "device is null";
  VulkanBuf2Image(device, filter, &filter_im, BufTYPE::NCHW);
  /*
        LOG(INFO)<<"conv_direct_w dims"<<param.filter->dims();
          const lite::vulkan::VulkanMem* filter_mem =
        param.filter->data<lite::vulkan::VulkanMem>();
      lite::vulkan::VulkanBuf* filter_vulkan_buf =
    (lite::vulkan::VulkanBuf*)filter_mem->mem.buf;
      float* pf = (float*)filter_vulkan_buf->mapped_ptr;
    for(int i = 0; i <filter_im.dims().production(); i++){
     // LOG(INFO)<<"conv_direct_wv:"<<i << " :"<<pf[i];
    }
  */
  // lite::Tensor tmp;
  // LOG(INFO)<<"=========================direct_conv_input"<<input_dim;
  // VulkanImage2Buf(device,input,&tmp, vulkan::BufTYPE::NCHW);
  // LOG(INFO)<<"=========================direct_conv_filter";
  // VulkanImage2Buf(device,&filter_im,&tmp, vulkan::BufTYPE::NCHW);

  // VulkanPrintImage(device, input);
  const lite::vulkan::VulkanMem* bias_param =
      bias->data<lite::vulkan::VulkanMem>();
  /*
    lite::vulkan::VulkanBuf* vulkan_buf =
    (lite::vulkan::VulkanBuf*)bias_param->mem.buf;
     LOG(INFO)<<"=========================direct_conv_bias";
     float* b_ptr = (float*)(vulkan_buf->mapped_ptr);
       for(int i = 0; i<bias->dims().production(); i++){
     // LOG(INFO)<<"conv_direct_bv:"<<i<<" :" <<b_ptr[i];
     }
  */
  // int bias_len4 = (filter_dim.data()[0]+3)/4 * 4;

  // if(b_data){
  //    bias_param =
  //    (lite::vulkan::VulkanMem*)TargetWrapper<TARGET(kVULKAN)>::Malloc(
  //           bias_len4 * sizeof(float));
  //           LOG(INFO)<<" param.bias->dims().production() * sizeof(float)"<<
  //           filter_dim.data()[0] * sizeof(float);
  //           LOG(INFO)<<" bias_len4 * sizeof(float)"<< bias_len4 *
  //           sizeof(float);
  //   CopySync<TARGET(kVULKAN)>(
  //       bias_param, b_data, filter_dim.data()[0] * sizeof(float),
  //       IoDirection::HtoD);

  // }

  std::vector<VkDescriptorType> types{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                      VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                      VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                      VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER};
  device->CreateDescriptorSetLayout(types);

  LOG(INFO) << "vulkan_conv_d";
  device->CreatePipelineLayout(device->setLayout);
  std::string conv = "conv_direct";
  if (fuse_relu) {
    conv += "_RELU";
  }
  device->CreateComputePipelines(conv);

  device->CreateDescriptorPool(1);
  device->AllocateDescriptorSets(*(device->descriptorpool), device->setLayout);
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

  const lite::vulkan::VulkanMem* filter_im_mem =
      filter_im.data<lite::vulkan::VulkanMem>();
  int filter_index = 3;
  device->UpdateDescriptorSets(device->GetVkDevice(),
                               filter_im_mem,
                               filter_index,
                               types[filter_index],
                               VK_IMAGE_LAYOUT_GENERAL);

  int bias_index = 4;
  // const lite::vulkan::VulkanMem* biasim =
  //   bias_im.data<lite::vulkan::VulkanMem>();
  device->UpdateDescriptorSets(device->GetVkDevice(),
                               bias_param,
                               bias_index,
                               types[bias_index],
                               VK_IMAGE_LAYOUT_GENERAL);

  LOG(INFO) << "vulkan_conv_d";
  device->AllocateCommandBuffers(&cmd);
  LOG(INFO) << "vulkan_conv_d";
  device->BeginCommandBuffer(cmd, 0);

  LOG(INFO) << "vulkan_conv_d";
  // LOG(INFO)<<"vkCmdBindPipeline !!!!\n";
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, device->pipeline);

  LOG(INFO) << "vulkan_conv_d";
  // Bind descriptor set.
  vkCmdBindDescriptorSets(cmd,
                          VK_PIPELINE_BIND_POINT_COMPUTE,
                          device->pipelineLayout,
                          0,
                          1,
                          device->descriptorSet,
                          0,
                          nullptr);

  LOG(INFO) << "vulkan_conv_d";
  VLOG(5) << "vkCmdBindDescriptorSets!!!!";

  vkCmdDispatch(
      cmd, (out_image_dim[0] + 15) / 16, (out_image_dim[1] + 15) / 16, 1);
  LOG(INFO) << "vulkan_conv_d";

  device->EndCommandBuffer(cmd);
  /*
    vkResetFences(device->GetVkDevice(), 1, &device->fence);
    device->QueueSubmit(device->compute_queue, cmd, device->fence);

    VkResult res = vkWaitForFences(
        device->GetVkDevice(), 1, &device->fence, VK_TRUE, UINT64_MAX);
    if (res) LOG(FATAL) << "vkWaitForFences error";
  */
  /*
     // VulkanPrintImage(device,output,BufTYPE::NCHW);
     Tensor *in = new Tensor();

             VulkanImage2Buf(device,output,in, vulkan::BufTYPE::NCHW);
             auto ptr = in->data<float>();
             float sum =0;
             for (int i = 0; i < in->numel(); ++i) {
               sum += ptr[i];
             //   LOG(INFO)<<"conv_direct_ov:"<<i << " :"<<ptr[i];

             }
             float output_agv =  sum / in->numel();
             LOG(INFO)<<"-----------------conv_direct_ovagv---"<<output_dim<<"-------------:"<<output_agv;



     //           LOG(INFO)<<"conv_direct_x dims"<<param.x->dims();
     // for(int i = 0; i < param.x->dims().production(); i++){
     //   LOG(INFO)<<"conv_direct_x:"<<i << " :"<<i_data[i];
     // }
     Tensor *ff = new Tensor();

             VulkanImage2Buf(device,&filter_im,ff, vulkan::BufTYPE::NCHW);
             auto pff = ff->data<float>();

     //   LOG(INFO)<<"conv_direct_w dims"<<param.filter->dims();
     // for(int i = 0; i <filter_im.dims().production(); i++){
     //   LOG(INFO)<<"conv_direct_wv:"<<i << " :"<<pff[i];
     // }
   */
}
void DirectConv::Run() { device->cmds.emplace_back(cmd); }

}  // namespace vulkan
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
