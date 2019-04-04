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

#include "utils/opencv_mp4_writer.h"

#include <sstream>
#include <stdexcept>
#include <string>

constexpr auto NAME = "OpenCvMp4Writer";
constexpr auto SOURCE_NAME = "input";
constexpr auto SINK_NAME = "output";

OpenCvMp4Writer::OpenCvMp4Writer(const std::string& field, int fps,
                                 const std::string& output_filepath)
    : Operator(OPERATOR_TYPE_CUSTOM, {SOURCE_NAME}, {SINK_NAME}),
      field_(field),
      fps_(fps),
      output_filepath_(output_filepath) {}

std::shared_ptr<OpenCvMp4Writer> OpenCvMp4Writer::Create(
    const FactoryParamsType&) {
  SAF_NOT_IMPLEMENTED;
  return nullptr;
}

std::string OpenCvMp4Writer::GetName() const { return NAME; }

void OpenCvMp4Writer::SetSource(StreamPtr stream) {
  Operator::SetSource(SOURCE_NAME, stream);
}

StreamPtr OpenCvMp4Writer::GetSink() { return Operator::GetSink(SINK_NAME); }

bool OpenCvMp4Writer::Init() { return true; }

bool OpenCvMp4Writer::OnStop() {
  writer_->release();
  return true;
}

void OpenCvMp4Writer::Process() {
  std::unique_ptr<Frame> frame = GetFrame(SOURCE_NAME);
  cv::Mat img = frame->GetValue<cv::Mat>(field_);

  // The writer is created when we get the first frame because we need to know
  // the dimensions of a frame to create the writer.
  if (writer_ == nullptr) {
    writer_ = std::make_unique<cv::VideoWriter>(
        output_filepath_, cv::CAP_FFMPEG | cv::CAP_MODE_BGR,
        cv::VideoWriter::fourcc('H', '2', '6', '4'), fps_,
        cv::Size(img.rows, img.cols), true);
  }

  writer_->write(img);
  PushFrame(SINK_NAME, std::move(frame));
};
