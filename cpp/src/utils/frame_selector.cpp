// Copyright 2018 The FilterForward Authors. All Rights Reserved.
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

#include "utils/frame_selector.h"

#include <memory>

constexpr auto NAME = "FrameSelector";
constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";

FrameSelector::FrameSelector(const std::string& conf_filepath)
    : Operator(OPERATOR_TYPE_CUSTOM, {SOURCE_NAME}, {SINK_NAME}) {
  std::ifstream conf_file(conf_filepath);
  std::string line;
  while (std::getline(conf_file, line)) {
    frame_ids_.insert(std::stoul(line));
  }
}

std::shared_ptr<FrameSelector> FrameSelector::Create(const FactoryParamsType&) {
  SAF_NOT_IMPLEMENTED;
  return nullptr;
}

std::string FrameSelector::GetName() const { return NAME; }

void FrameSelector::SetSource(StreamPtr stream) {
  Operator::SetSource(SOURCE_NAME, stream);
}

StreamPtr FrameSelector::GetSink() { return Operator::GetSink(SINK_NAME); }

bool FrameSelector::Init() { return true; }

bool FrameSelector::OnStop() { return true; }

void FrameSelector::Process() {
  std::unique_ptr<Frame> frame = GetFrame(SOURCE_NAME);
  LOG(INFO) << frame->GetValue<unsigned long>(Frame::kFrameIdKey);
  if (frame_ids_.find(frame->GetValue<unsigned long>(Frame::kFrameIdKey)) !=
      frame_ids_.end()) {
    LOG(INFO) << "Pushing frame "
              << frame->GetValue<unsigned long>(Frame::kFrameIdKey);
    PushFrame(SINK_NAME, std::move(frame));
  }
};
