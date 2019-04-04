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

#ifndef FILTERFORWARD_UTILS_FRAME_SELECTOR_H_
#define FILTERFORWARD_UTILS_FRAME_SELECTOR_H_

#include <memory>
#include <string>
#include <unordered_set>

#include "common/types.h"
#include "operator/operator.h"

class FrameSelector : public Operator {
 public:
  FrameSelector(const std::string& conf_filepath);

  static std::shared_ptr<FrameSelector> Create(const FactoryParamsType& params);

  virtual std::string GetName() const override;

  void SetSource(StreamPtr stream);
  using Operator::SetSource;

  StreamPtr GetSink();
  using Operator::GetSink;

 protected:
  virtual bool Init() override;
  virtual bool OnStop() override;
  virtual void Process() override;

 private:
  std::unordered_set<unsigned long> frame_ids_;
};

#endif  // FILTERFORWARD_UTILS_FRAME_SELECTOR_H_
