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

#include "ff/ed.h"

#include <boost/date_time/posix_time/posix_time.hpp>
#include "saf.h"

#include "ff/imagematch.h"

constexpr auto NAME = "Ed";
constexpr auto SINK_NAME = "output";
constexpr auto SOURCE_NAME = "input";

const char* Ed::kEventIdKey = "event_id";
const char* Ed::kKvMicrosKey = "kv_micros";
const char* Ed::kMcResultsKey = "mc_results";

Ed::Ed(std::vector<std::shared_ptr<Filter>> filters, unsigned long n,
       bool vote_for_whole_buf, bool mc_pass_all)
    : Operator(OPERATOR_TYPE_CUSTOM, {SOURCE_NAME}, {SINK_NAME}),
      buf_{n},
      filters_(filters),
      vote_for_whole_buf_(vote_for_whole_buf),
      mc_pass_all_(mc_pass_all) {
  CHECK(n % 2 == 1) << "n must be odd, but is: " << n;
}

std::shared_ptr<Ed> Ed::Create(const FactoryParamsType&) {
  SAF_NOT_IMPLEMENTED;
  return nullptr;
}

void Ed::SetSource(StreamPtr stream) {
  Operator::SetSource(SOURCE_NAME, stream);
}

StreamPtr Ed::GetSink() { return Operator::GetSink(SINK_NAME); }

std::string Ed::GetName() const { return NAME; }

bool Ed::Init() { return true; }

bool Ed::OnStop() { return true; }

void Ed::Process() {
  auto input_frame = GetFrame(SOURCE_NAME);
  buf_.push_back(std::move(input_frame));
  // Wait until the frame buffer has at least k frames to do k voting
  if (!buf_.full()) {
    return;
  }

  // Perform k voting for all filters.
  boost::posix_time::ptime kv_start_micros =
      boost::posix_time::microsec_clock::local_time();
  // Maps filter ID to whether that filter had enough votes to pass.
  std::unordered_map<int, bool> mc_results;
  for (auto& filter : filters_) {
    int votes = 0;
    for (auto& frame : buf_) {
      if (frame->GetValue<std::unordered_map<int, bool>>(
              ImageMatch::kMatchesKey)[filter->id_]) {
        ++votes;
      }
    }

    mc_results[filter->id_] = votes >= filter->kv_k_;
  }

  // Collect the frames that we are going to apply the K-Voting results to.
  std::vector<std::unique_ptr<Frame>> output_frames;
  if (vote_for_whole_buf_) {
    // Apply the K-Voting result to the entire window.
    for (auto& frame : buf_) {
      output_frames.push_back(std::move(frame));
    }
    buf_.clear();
  } else {
    // Apply the K-Voting result to the middle frame only.
    output_frames.push_back(std::make_unique<Frame>(buf_[buf_.size() / 2 + 1]));
  }

  // Store K-Voting results in the output frames.
  for (auto& output_frame : output_frames) {
    std::unordered_map<unsigned long, int> event_id_map;
    for (auto& filter : filters_) {
      int id = filter->id_;
      bool mc_result = mc_results[id];
      int event_id = filter->EventTransition(mc_result);
      if (mc_result) {
        event_id_map[id] = event_id;
      }
    }
    output_frame->SetValue(kEventIdKey, event_id_map);
  }

  boost::posix_time::time_duration kv_micros =
      boost::posix_time::microsec_clock::local_time() - kv_start_micros;

  for (auto& frame : output_frames) {
    frame->SetValue(kKvMicrosKey, kv_micros);
    frame->SetValue(Ed::kMcResultsKey, mc_results);

    bool push_frame = mc_pass_all_;
    if (!push_frame) {
      // If we are not support to automatically push all frames, then look
      // through the MC results to see if any of the MCs activated on this
      // frame. If at least one activated, then we need to push the frame.
      for (const auto& pair : mc_results) {
        if (pair.second) {
          push_frame = true;
          // As soon as we find one MC that activated, then we know we nede to
          // push this frame.
          break;
        }
      }
    }

    if (push_frame) {
      PushFrame(SINK_NAME, std::move(frame));
    } else {
      // We are discarding this frame, so release its flow control token.
      FlowControlEntrance* fce = frame->GetFlowControlEntrance();
      if (fce != nullptr) {
        fce->ReturnToken(frame->GetValue<unsigned long>(Frame::kFrameIdKey));
      }
    }
  }
}
