/* Copyright (c) 2016 paddlepaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <algorithm>
#include <vector>
#include "lite/backends/cuda/math/pooling.h"
//#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

template <typename PoolProcess, typename T>
__global__ void KernelPool2D(const int nthreads, const T* input_data,
                             const int channels, const int input_height,
                             const int input_width, const int output_height,
                             const int output_width, const int ksize_height,
                             const int ksize_width, const int stride_height,
                             const int stride_width, const int padding_height,
                             const int padding_width, PoolProcess pool_process,
                             bool exclusive, bool adaptive, T* output_data) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < nthreads;
       index += blockDim.x * gridDim.x) {
    int pw = index % output_width;
    int ph = (index / output_width) % output_height;
    int c = (index / output_width / output_height) % channels;
    int batch_idx = index / output_width / output_height / channels;

    int hstart, hend;
    int wstart, wend;
    if (adaptive) {
      hstart = AdaptStartIndex(ph, input_height, output_height);
      hend = AdaptEndIndex(ph, input_height, output_height);

      wstart = AdaptStartIndex(pw, input_width, output_width);
      wend = AdaptEndIndex(pw, input_width, output_width);
    } else {
      hstart = ph * stride_height - padding_height;
      hend = min(hstart + ksize_height, input_height);
      hstart = max(hstart, 0);

      wstart = pw * stride_width - padding_width;
      wend = min(wstart + ksize_width, input_width);
      wstart = max(wstart, 0);
    }

    input_data += (batch_idx * channels + c) * input_height * input_width;
    T ele = pool_process.initial();
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        pool_process.compute(input_data[h * input_width + w], &ele);
      }
    }
    int pool_size = (exclusive || adaptive) ? (hend - hstart) * (wend - wstart)
                                            : ksize_height * ksize_width;
    pool_process.finalize(static_cast<T>(pool_size), &ele);
    output_data[index] = ele;
  }
}


/*
 * All tensors are in NCHW format.
 * Ksize, strides, paddings are two elements. These two elements represent
 * height and width, respectively.
 */
template <typename PoolProcess, typename T>
class Pool2dFunctor<lite::TargetType::kCUDA, PoolProcess, T> {
 public:
  //void operator()(const lite::CUDAContext& context,
  void operator()(){

/*
  void operator()(
                  lite::Tensor& input, const std::vector<int>& ksize,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings, PoolProcess pool_process,
                  bool exclusive, bool adaptive, lite::Tensor* output) {
    const int batch_size = input.dims()[0];
    const int input_channels = input.dims()[1];
    const int input_height = input.dims()[2];
    const int input_width = input.dims()[3];
    const int output_channels = output->dims()[1];
    const int output_height = output->dims()[2];
    const int output_width = output->dims()[3];
    const int ksize_height = ksize[0];
    const int ksize_width = ksize[1];
    const int stride_height = strides[0];
    const int stride_width = strides[1];
    const int padding_height = paddings[0];
    const int padding_width = paddings[1];

    const T* input_data = input.data<T>();
    T* output_data = output->mutable_data<T>(TARGET(kCUDA));

    int nthreads = batch_size * output_channels * output_height * output_width;
    int blocks = (nthreads + 1024 - 1) / 1024;
    dim3 threads(1024, 1);
    dim3 grid(blocks, 1);

    KernelPool2D<PoolProcess, T><<<grid, threads, 0,  context.exec_stream()>>>(
        nthreads, input_data, input_channels, input_height, input_width,
        output_height, output_width, ksize_height, ksize_width, stride_height,
        stride_width, padding_height, padding_width, pool_process, exclusive,
        adaptive, output_data);
  */
  }
};



/*

template <typename PoolProcess, typename T>
void Pool2dDirectCUDAFunctor<PoolProcess, T>::operator()(
    const T* input, const std::vector<int>& input_shape,
    const std::vector<int>& output_shape, const std::vector<int>& ksize,
    const std::vector<int>& strides, const std::vector<int>& paddings,
    PoolProcess pool_compute, bool exclusive, T* output, cudaStream_t stream) {
  const int batch_size = input_shape[0];
  const int input_channels = input_shape[1];
  const int input_height = input_shape[2];
  const int input_width = input_shape[3];
  const int output_channels = output_shape[1];
  const int output_height = output_shape[2];
  const int output_width = output_shape[3];
  const int ksize_height = ksize[0];
  const int ksize_width = ksize[1];
  const int stride_height = strides[0];
  const int stride_width = strides[1];
  const int padding_height = paddings[0];
  const int padding_width = paddings[1];

  int nthreads = batch_size * output_channels * output_height * output_width;
  int blocks = (nthreads + 1024 - 1) / 1024;
  dim3 threads(1024, 1);
  dim3 grid(blocks, 1);

  KernelPool2D<PoolProcess, T><<<grid, threads, 0, stream>>>(
      nthreads, input, input_channels, input_height, input_width, output_height,
      output_width, ksize_height, ksize_width, stride_height, stride_width,
      padding_height, padding_width, pool_compute, exclusive, false, output);
}
*/
template class Pool2dFunctor<lite::TargetType::kCUDA,
                             lite::cuda::math::MaxPool<float>, float>;
template class Pool2dFunctor<lite::TargetType::kCUDA,
                             lite::cuda::math::AvgPool<float>, float>;

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle

