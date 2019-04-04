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

#ifndef FILTERFORWARD_UTILS_OPENCV_MP4_WRITER_H_
#define FILTERFORWARD_UTILS_OPENCV_MP4_WRITER_H_

#include <memory>
#include <string>

#include <opencv2/opencv.hpp>
#include "saf.h"

class OpenCvMp4Writer : public Operator {
 public:
  OpenCvMp4Writer(const std::string& field, int fps,
                  const std::string& output_filepath);

  static std::shared_ptr<OpenCvMp4Writer> Create(
      const FactoryParamsType& params);

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
  // The frame field that will be encoded.
  std::string field_;
  int fps_;
  const std::string& output_filepath_;
  std::unique_ptr<cv::VideoWriter> writer_;
};

#endif  // FILTERFORWARD_UTILS_OPENCV_MP4_WRITER_H_
