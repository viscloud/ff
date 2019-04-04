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

#ifndef FILTERFORWARD_FILTERFORWARD_ED_H_
#define FILTERFORWARD_FILTERFORWARD_ED_H_

#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <boost/circular_buffer.hpp>
#include "saf.h"

#include "ff/filter.h"

// A operator that detects events using k-voting. Expects ImageMatch metadata
// to be in the input frames
class Ed : public Operator {
 public:
  Ed(std::vector<std::shared_ptr<Filter>> filters, unsigned long n,
     bool vote_for_whole_buf = false, bool mc_pass_all = false);
  static std::shared_ptr<Ed> Create(const FactoryParamsType& params);

  void SetSource(StreamPtr stream);
  using Operator::SetSource;

  StreamPtr GetSink();
  using Operator::GetSink;

  std::string GetName() const override;

  static const char* kEventIdKey;
  static const char* kKvMicrosKey;
  static const char* kMcResultsKey;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  boost::circular_buffer<std::unique_ptr<Frame>> buf_;
  std::vector<std::shared_ptr<Filter>> filters_;
  // If true, K-Voting results will be applied to the entire window, otherwise
  // the results will be applied to only the middle frame.
  bool vote_for_whole_buf_;
  bool mc_pass_all_;
};

#endif  // FILTERFORWARD_FILTERFORWARD_ED_H_
