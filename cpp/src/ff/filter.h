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

#ifndef FILTERFORWARD_FILTERFORWARD_FILTER_H_
#define FILTERFORWARD_FILTERFORWARD_FILTER_H_

#include "ff/fv_gen.h"

#include <tensorflow/core/public/session.h>

class Filter {
 public:
  Filter(int id, const FvSpec& fv_spec, const std::string& classifier_path,
         const std::string& mc_input_name, const std::string& mc_output_name,
         int mc_num_outputs, float threshold, int kv_k, size_t max_buf_len,
         long iff_timeout_millis);
  std::string ToString();
  int EventTransition(bool cur_frame_is_event);

  int id_;
  FvSpec fv_spec_;
  std::string classifier_path_;
  std::string mc_input_name_;
  std::string mc_output_name_;
  int mc_num_outputs_;
  tensorflow::Session* classifier_;
  float threshold_;
  int kv_k_;
  size_t max_buf_len_;
  long iff_timeout_millis_;
  int cur_event_id_;
  bool prev_frame_was_event_;
  bool prev_prev_frame_was_event_;
};

#endif  // FILTERFORWARD_FILTERFORWARD_FILTER_H_
