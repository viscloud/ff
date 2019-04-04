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

#include "ff/filter.h"

Filter::Filter(int id, const FvSpec& fv_spec,
               const std::string& classifier_path,
               const std::string& mc_input_name,
               const std::string& mc_output_name, int mc_num_outputs,
               float threshold, int kv_k, size_t max_buf_len,
               long iff_timeout_millis)
    : id_(id),
      fv_spec_(fv_spec),
      classifier_path_(classifier_path),
      mc_input_name_(mc_input_name),
      mc_output_name_(mc_output_name),
      mc_num_outputs_(mc_num_outputs),
      threshold_(threshold),
      kv_k_(kv_k),
      max_buf_len_(max_buf_len),
      iff_timeout_millis_(iff_timeout_millis),
      cur_event_id_(0),
      prev_frame_was_event_(false),
      prev_prev_frame_was_event_(false) {
  tensorflow::GraphDef graph_def;
  tensorflow::Status status =
      ReadBinaryProto(tensorflow::Env::Default(), classifier_path, &graph_def);
  if (!status.ok()) {
    LOG(FATAL) << "Failed to load TensorFlow graph: " << status.error_message();
  }

  classifier_ = tensorflow::NewSession(tensorflow::SessionOptions());
  status = classifier_->Create(graph_def);
  if (!status.ok()) {
    LOG(FATAL) << "Failed to create TensorFlow Session: "
               << status.error_message();
  }
}

std::string Filter::ToString() {
  std::ostringstream summary;
  summary << "Filter(ID: " << id_ << ", FvSpec: " << fv_spec_.ToString()
          << ", Classifier: " << classifier_path_
          << ", Threshold: " << threshold_ << ", K-Voting K: " << kv_k_
          << ", Max IFF Buf Len: " << max_buf_len_
          << ", IFF Timeout Millis: " << iff_timeout_millis_ << ")";
  return summary.str();
}

int Filter::EventTransition(bool cur_frame_is_event) {
  if (cur_frame_is_event != prev_frame_was_event_) {
    if (prev_frame_was_event_) {
      ++cur_event_id_;
    }
    prev_prev_frame_was_event_ = prev_frame_was_event_;
    prev_frame_was_event_ = cur_frame_is_event;
  }
  return cur_event_id_;
}
