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

#include "lite/core/kernel.h"
#include <cstdlib>
#include "lite/utils/string.h"
#define RUN_ON 0
inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}
static float all_time_gemmlike=0;
static float all_time_gemmlike_v=0;
static float SaberEltwise=0;
static float SaberConv2D=0;
static float SaberSoftmax=0;
static float SaberScale=0;
static float SaberSplit=0;
static float SaberShuffleChannel=0;
static float SaberConcat=0;
static float SaberActivation=0;
static float SaberSlice=0;
static float SaberDeconv2D=0;
static float SaberFlatten=0;
static float SaberPriorBox=0;
static float SaberReshape=0;
static float SaberPooling=0;
static float _box_coder=0;
static float _multiclass_nms=0;

static float SaberDetectionOutput=0;
static float SaberFc=0;
static float _n_time=0;
static int cnt_gemmlike = 0;
  int op_nums[] = {58 /* 58 59 71*/,71, 18, 56, 141 /*121*/,66/*57*/, 19,0,0,112, 73,142/* 137 */,62/*72 */, 65/* 75 */,0,0,142/*141 no conv_conv*/};
    int OP_num = op_nums[11];
namespace paddle {
namespace lite {
  void KernelBase::Launch() {
    /// First run, init kernel, do weights transform once
    if (is_first_epoch_) {
      PrepareForRun();
      is_first_epoch_ = false;
    }
    /// re-init the kernel if needed (input shape should be checked in conv
    /// kernel)
    ReInitWhenNeeded();

    // Reset the workspace to make every kernel in the same thread to share the
    // temporary memory.
    WorkSpace::Global_Host().AllocReset();
#if defined(LITE_WITH_X86)
    WorkSpace::Global_X86().AllocReset();
#endif
#if defined(LITE_WITH_CUDA)
    WorkSpace::Global_CUDA().AllocReset();
#endif
#if defined(LITE_WITH_MLU)
    WorkSpace::Global_MLU().AllocReset();
#endif
#ifdef LITE_WITH_PROFILE
    profiler_->StopTiming(profile::Type::kCreate, profile_id_, ctx_.get());
    profiler_->StartTiming(profile::Type::kDispatch, profile_id_, ctx_.get());
    Run();

    if (is_first_epoch_for_profiler_) {
      SetProfileRuntimeKernelInfo(profiler_->GetOpCharacter(profile_id_));
      is_first_epoch_for_profiler_ = false;
    }
    profiler_->StopTiming(profile::Type::kDispatch, profile_id_, ctx_.get());

#else
// LOG(INFO)<<"Lauch:"<<l_cnt++<<"   "<<name();
#if RUN_ON
  double start_v = 0;
  double end_v = 0;
  double duration_v = 0;

  auto start = GetCurrentUS();
       start_v = GetCurrentUS();

    Run();
      end_v = GetCurrentUS();

            auto duration = (GetCurrentUS() - start) / 1000.0;
    cnt_gemmlike++;
      duration_v = (end_v - start_v) / 1000.0;
      std::string name_ =name();
        // LOG(INFO) << "name_:" << name_<< "      cnt:"<<cnt_gemmlike;
    const char* cname = name_.c_str();

    if(cnt_gemmlike > 10 * OP_num ){
      all_time_gemmlike+= duration;
      all_time_gemmlike_v+= duration_v;

      if(strstr(cname, "conv2d:arm/")){
        SaberConv2D += duration_v;

      }else if(strstr(cname, "elementwise")){
        SaberEltwise += duration_v;

      } else if (strstr(cname, "softmax")) {
        SaberSoftmax += duration_v;

      } else if (strstr(cname, "scale")) {
        SaberScale += duration_v;

      }else if (strstr(cname, "pool2d")) {
        SaberPooling += duration_v;

      }else if (strstr(cname, "split")) {
        SaberSplit += duration_v;

      }else if (strstr(cname, "shuffle_channel")) {
        SaberShuffleChannel += duration_v;

      }else if (strstr(cname, "concat")) {
        SaberConcat += duration_v;

      }else if (strstr(cname, "SaberActivation")) {
        SaberActivation += duration_v;

      }else if (strstr(cname, "slice")) {
        SaberSlice += duration_v;

      }else if (strstr(cname, "conv2d_transpose")) {
        SaberDeconv2D += duration_v;

      }else if (strstr(cname, "transpose2")) {
        SaberFlatten += duration_v;

      }else if (strstr(cname, "prior_box")) {
        SaberPriorBox += duration_v;

      }else if (strstr(cname, "reshape")) {
        SaberReshape += duration_v;

      }else if (strstr(cname, "calib")) {
        SaberDetectionOutput += duration_v;

      }else if (strstr(cname, "fc")) {
        SaberFc += duration_v;

      }else if (strstr(cname, "box_coder")) {
        _box_coder += duration_v;

      }else if (strstr(cname, "multiclass_nms")) {
        _multiclass_nms += duration_v;

      }else {
        _n_time += duration_v;
        // LOG(INFO) << "name_:" << name_<< "      cnt:"<<cnt_gemmlike;
        
      }
    }
    if( cnt_gemmlike >= 1010 * OP_num){
        LOG(INFO) << "cnt_gemmlike:" <<cnt_gemmlike << " " << all_time_gemmlike<< " ms";
        LOG(INFO) << "all_time_gemmlike_v:" << all_time_gemmlike_v<< " ms";
        LOG(INFO) << "SaberConv2D:" << SaberConv2D<< " ms";
        LOG(INFO) << "SaberEltwise:" << SaberEltwise<< " ms";
        LOG(INFO) << "SaberSoftmax:" << SaberSoftmax<< " ms";
        LOG(INFO) << "SaberScale:" << SaberScale<< " ms";
        LOG(INFO) << "SaberPooling:" << SaberPooling<< " ms";
        LOG(INFO) << "SaberSplit:" << SaberSplit<< " ms";
        LOG(INFO) << "SaberShuffleChannel:" << SaberShuffleChannel<< " ms";
        LOG(INFO) << "SaberConcat:" << SaberConcat<< " ms";
        LOG(INFO) << "SaberActivation:" << SaberActivation<< " ms";
        LOG(INFO) << "SaberSlice:" << SaberSlice<< " ms";
        LOG(INFO) << "SaberDeconv2D:" << SaberDeconv2D<< " ms";
        LOG(INFO) << "SaberFlatten:" << SaberFlatten<< " ms";
        LOG(INFO) << "SaberPriorBox:" << SaberPriorBox<< " ms";
        LOG(INFO) << "SaberReshape:" << SaberReshape<< " ms";
        LOG(INFO) << "calib:" << SaberDetectionOutput<< " ms";
        LOG(INFO) << "SaberFc:" << SaberFc<< " ms";
        LOG(INFO) << "_box_coder:" << _box_coder<< " ms";
        LOG(INFO) << "_multiclass_nms:" << _multiclass_nms<< " ms";

        LOG(INFO) << "others:" << _n_time<< " ms";

    }
  #else
    Run();

  #endif
#endif
  }
std::string KernelBase::summary() const {
  STL::stringstream ss;
  ss << op_type() << ":" << TargetToStr(target()) << "/"
     << PrecisionToStr(precision()) << "/" << DataLayoutToStr(layout()) << "("
     << alias() << ")";
  return ss.str();
}

const Type *KernelBase::GetInputDeclType(const std::string &arg_name) const {
  CHECK(!op_type_.empty()) << "op_type should be set first";
  const auto *type = ParamTypeRegistry::Global().RetrieveInArgument(
      place(), GenParamTypeKey(), arg_name);
  CHECK(type) << "no type registered for kernel [" << op_type_
              << "] input argument [" << arg_name << "]"
              << " with key " << GenParamTypeKey();
  return type->type;
}

const Type *KernelBase::GetOutputDeclType(const std::string &arg_name) const {
  CHECK(!op_type_.empty()) << "op_type should be set first";
  const auto *type = ParamTypeRegistry::Global().RetrieveOutArgument(
      place(), GenParamTypeKey(), arg_name);
  CHECK(type) << "no type registered for kernel [" << GenParamTypeKey()
              << "] output argument [" << arg_name << "]";
  return type->type;
}

std::string KernelBase::GenParamTypeKey() const {
  STL::stringstream ss;
  ss << op_type() << "/" << alias_;
  return ss.str();
}

void KernelBase::ParseKernelType(const std::string &kernel_type,
                                 std::string *op_type,
                                 std::string *alias,
                                 Place *place) {
  auto parts = lite::SplitView(kernel_type, '/');
  CHECK_EQ(parts.size(), 5u);

  *op_type = parts[0];
  *alias = parts[1];

  const auto &target = parts[2];
  const auto &precision = parts[3];
  const auto &layout = parts[4];

  place->target = static_cast<TargetType>(target.to_digit<int>());
  place->precision = static_cast<PrecisionType>(precision.to_digit<int>());
  place->layout = static_cast<DataLayoutType>(layout.to_digit<int>());
}

std::string KernelBase::SerializeKernelType(const std::string &op_type,
                                            const std::string &alias,
                                            const Place &place) {
  STL::stringstream ss;
  ss << op_type << "/";
  ss << alias << "/";
  // We serialize the place value not the string representation here for
  // easier deserialization.
  ss << static_cast<int>(place.target) << "/";
  ss << static_cast<int>(place.precision) << "/";
  ss << static_cast<int>(place.layout);
  return ss.str();
}

bool ParamTypeRegistry::KeyCmp::operator()(
    const ParamTypeRegistry::key_t &a,
    const ParamTypeRegistry::key_t &b) const {
  return a.hash() < b.hash();
}

STL::ostream &operator<<(STL::ostream &os,
                         const ParamTypeRegistry::KernelIdTy &other) {
  std::string io_s = other.io == ParamTypeRegistry::IO::kInput ? "in" : "out";
  os << other.kernel_type << ":" << other.arg_name << ":" << io_s << ":"
     << other.place.DebugString();
  return os;
}

}  // namespace lite
}  // namespace paddle
