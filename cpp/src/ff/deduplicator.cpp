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

#include "ff/deduplicator.h"

constexpr auto NAME = "Deduplicator";
constexpr auto SINK_NAME = "output";
constexpr auto SOURCE_NAME = "input";

Deduplicator::Deduplicator(std::unordered_set<std::string> fields)
    : Operator(OPERATOR_TYPE_CUSTOM, {SOURCE_NAME}, {SINK_NAME}),
      fields_(fields) {}

std::shared_ptr<Deduplicator> Deduplicator::Create(const FactoryParamsType&) {
  SAF_NOT_IMPLEMENTED;
  return nullptr;
}

void Deduplicator::SetSource(StreamPtr stream) {
  Operator::SetSource(SOURCE_NAME, stream);
}

StreamPtr Deduplicator::GetSink() { return Operator::GetSink(SINK_NAME); }

std::string Deduplicator::GetName() const { return NAME; }

bool Deduplicator::Init() { return true; }

bool Deduplicator::OnStop() { return true; }

void Deduplicator::Process() {
  std::unique_ptr<Frame> frame = GetFrame(SOURCE_NAME);
  unsigned long frame_id = frame->GetValue<unsigned long>(Frame::kFrameIdKey);
  if (sent_frame_ids_.find(frame_id) == sent_frame_ids_.end()) {
    sent_frame_ids_.insert(frame_id);
    PushFrame(SINK_NAME, std::move(frame));
  } else {
    PushFrame(SINK_NAME, std::make_unique<Frame>(frame, fields_));
  }
}
