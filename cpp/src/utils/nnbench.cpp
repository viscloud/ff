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

#include "utils/nnbench.h"

constexpr auto NAME = "NnBench";
constexpr auto SINK_NAME = "output";
constexpr auto SOURCE_NAME = "input";

NnBench::NnBench(const ModelDesc& model_desc, const Shape& input_shape,
                 int batch_size, int num_copies)
    : Operator(OPERATOR_TYPE_CUSTOM, {SOURCE_NAME}, {SINK_NAME}),
      batch_size_(batch_size),
      input_layer_(model_desc.GetDefaultInputLayer()),
      output_layer_(model_desc.GetDefaultOutputLayer()) {
  // Load models.
  auto& manager = ModelManager::GetInstance();
  for (decltype(num_copies) i = 0; i < num_copies; ++i) {
    std::unique_ptr<Model> model =
        manager.CreateModel(model_desc, input_shape, batch_size_);
    model->Load();
    models_.push_back(std::move(model));
  }
}

std::shared_ptr<NnBench> NnBench::Create(const FactoryParamsType&) {
  SAF_NOT_IMPLEMENTED;
  return nullptr;
}

std::string NnBench::GetName() const { return NAME; }

bool NnBench::Init() { return true; }

bool NnBench::OnStop() { return true; }

void NnBench::SetSource(StreamPtr stream) { SetSource(SOURCE_NAME, stream); }

StreamPtr NnBench::GetSink() { return Operator::GetSink(SINK_NAME); }

void NnBench::Process() {
  std::unique_ptr<Frame> input_frame = GetFrame(SOURCE_NAME);
  // Do the conversion before pushing the frame to the batch buffer to minimize
  // the amount of work that must be done when the buffer fills up.
  std::string image_key = GetName() + ".image.normalized";
  input_frame->SetValue(image_key,
                        models_.front()->ConvertAndNormalize(
                            input_frame->GetValue<cv::Mat>("image")));

  buf_.push_back(std::move(input_frame));
  if (buf_.size() < batch_size_) {
    return;
  }

  std::vector<cv::Mat> batch;
  for (auto& frame : buf_) {
    batch.push_back(frame->GetValue<cv::Mat>(image_key));
  }

  for (auto& model : models_) {
    model->Evaluate({{input_layer_, batch}}, {output_layer_});
  }

  // Push all of the frames out the sink.
  for (auto& output_frame : buf_) {
    PushFrame(SINK_NAME, std::move(output_frame));
  }

  buf_.clear();
}
