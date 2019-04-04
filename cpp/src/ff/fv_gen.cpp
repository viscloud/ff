// Copyright 2016 The FilterForward Authors. All Rights Reserved.
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

#include "ff/fv_gen.h"

constexpr auto NAME = "FvGen";
constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";

FvSpec::FvSpec(const std::string& layer_name, int xmin, int ymin, int xmax,
               int ymax, bool flat)
    : layer_name_(layer_name),
      roi_(xmin, ymin, xmax - xmin, ymax - ymin),
      yrange_(ymin, ymax),
      xrange_(xmin, xmax),
      xmin_(xmin),
      xmax_(xmax),
      ymin_(ymin),
      ymax_(ymax),
      flat_(flat) {
  if (ymin == 0 && ymax == 0) {
    LOG(INFO) << "No bounds specified for Feature Vector vertical axis, "
                 "using full output";
    yrange_ = cv::Range::all();
  }
  if (xmin == 0 && xmax == 0) {
    LOG(INFO) << "No bounds specified for Feature Vector horizontal axis, "
                 "using full output";
    xrange_ = cv::Range::all();
  }
}

FvSpec::FvSpec(const FvSpec& other)
    : layer_name_(other.layer_name_),
      roi_(other.roi_),
      yrange_(other.yrange_),
      xrange_(other.xrange_),
      xmin_(other.xmin_),
      xmax_(other.xmax_),
      ymin_(other.ymin_),
      ymax_(other.ymax_),
      flat_(other.flat_) {}

std::string FvSpec::GetUniqueID() const {
  std::ostringstream ss;
  ss << layer_name_ << xmin_ << ymin_ << xmax_ << ymax_ << std::boolalpha
     << flat_;
  return ss.str();
}

std::string FvSpec::ToString() {
  std::ostringstream summary;
  summary << "FvSpec(Layer Name: " << layer_name_ << ", X Min: " << roi_.x
          << ", X Max: " << roi_.x + roi_.width << ", Y Min: " << roi_.y
          << ", Y Max: " << roi_.y + roi_.height << ")";
  return summary.str();
}

FvGen::FvGen() : Operator(OPERATOR_TYPE_CUSTOM, {SOURCE_NAME}, {SINK_NAME}) {}

std::shared_ptr<FvGen> FvGen::Create(const FactoryParamsType&) {
  SAF_NOT_IMPLEMENTED;
  return nullptr;
}

void FvGen::AddFv(const FvSpec& fv_spec) {
  feature_vector_specs_.push_back(fv_spec);
}

void FvGen::SetSource(StreamPtr stream) { SetSource(SOURCE_NAME, stream); }

StreamPtr FvGen::GetSink() { return Operator::GetSink(SINK_NAME); }

std::string FvGen::GetName() const { return NAME; }

bool FvGen::Init() { return true; }

bool FvGen::OnStop() { return true; }

void FvGen::Process() {
  std::unique_ptr<Frame> frame = GetFrame(SOURCE_NAME);

  for (const auto& spec : feature_vector_specs_) {
    cv::Mat input_mat = frame->GetValue<cv::Mat>(spec.layer_name_);
    frame->SetValue(spec.GetUniqueID(), input_mat);
  }

  PushFrame(SINK_NAME, std::move(frame));
}
